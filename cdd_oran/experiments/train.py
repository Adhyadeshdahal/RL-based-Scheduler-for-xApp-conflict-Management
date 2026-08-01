import logging
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from cdd_oran.config import DEFAULT_CONFIG, ExperimentConfig
from cdd_oran.conflicts import state_to_tensor
from cdd_oran.envs import get_env
from cdd_oran.models import get_model
from cdd_oran.policies.random_policy import RandomPolicy
from cdd_oran.utils.runs import create_run_dir, write_metrics
from cdd_oran.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class ReplayBufferDataset(Dataset):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s, a, s_next = self.data[index]
        return s, a, s_next

    def add(self, s, a, s_next):
        self.data.append((s, a, s_next))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.data), size=batch_size)
        s, a, s_next = zip(*(self.data[i] for i in idx), strict=True)
        return torch.stack(s), torch.tensor(a), torch.stack(s_next)


def main(cfg: ExperimentConfig = DEFAULT_CONFIG, resume=False, run_dir=None):
    seed_everything(cfg.seed, cfg.deterministic)
    env = get_env(cfg)

    ground_truth_causal_graph = env.true_adj_matrix
    run_dir = create_run_dir(cfg) if run_dir is None else run_dir
    checkpoint_path = run_dir / "checkpoint.pt"
    writer = SummaryWriter(run_dir / "tensorboard")

    state_dim = env.get_state_dim()
    action_dim = env.action_dim

    random_policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    model = get_model(cfg, env)

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_model(checkpoint_path)
        logger.info("Resumed from checkpoint: %s", checkpoint_path)

    train_buffer = ReplayBufferDataset(state_dim, action_dim)
    test_buffer = ReplayBufferDataset(state_dim, action_dim)
    obs = env.reset()
    loss = 0.0
    episode_reward = 0
    episode_rewards = []
    metrics: dict[str, Any] = {"prediction_mse": None}
    for step in range(cfg.train.total_steps):
        action = random_policy.act()

        next_obs, reward, done, info = env.step(action)

        s_tensor = state_to_tensor(obs)
        s_next_tensor = state_to_tensor(next_obs)
        train_buffer.add(
            s_tensor, action, s_next_tensor
        ) if random.random() > 0.2 else test_buffer.add(s_tensor, action, s_next_tensor)
        episode_reward += reward

        obs = next_obs
        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            logger.info("Episode %d reward: %s", len(episode_rewards), episode_rewards[-1])

            episode_reward = 0

        if step < cfg.train.init_steps:
            continue

        for _ in range(cfg.train.inference_gradient_steps):
            s, a, s_next = train_buffer.sample(cfg.model.batch_size)
            s = s.float().to(cfg.device)
            s_next = s_next.float().to(cfg.device)
            a = a.float().reshape(-1, action_dim).to(cfg.device)
            s_pair = torch.stack([s, s_next], dim=1).to(cfg.device)
            loss = model.train_step(s_pair, a)

        if cfg.model_kind == "cdl":
            if step % (cfg.train.eval_steps * cfg.train.inference_gradient_steps) == 0:
                s, a, s_next = train_buffer.sample(cfg.model.batch_size)
                s = s.float().to(cfg.device)
                s_next = s_next.float().to(cfg.device)
                a = a.float().reshape(-1, action_dim).to(cfg.device)
                s_pair = torch.stack([s, s_next], dim=1).to(cfg.device)
                model.update_mask(s_pair, a)

        if step % cfg.train.plot_freq == 0:
            if cfg.model_kind == "cdl":
                if len(test_buffer) >= cfg.train.test_batch_size:
                    s_b, a_b, s_1b = test_buffer.sample(cfg.train.test_batch_size)
                    s_b, a_b, s_1b = (
                        s_b.float().to(cfg.device),
                        a_b.float().reshape(-1, action_dim).to(cfg.device),
                        s_1b.float().to(cfg.device),
                    )
                    mse = model.evaluate_predictions(s_b, a_b, s_1b)
                    metrics["prediction_mse"] = mse
                    logger.info("Next-step prediction MSE: %s", mse)
                    writer.add_scalar("Predictions/MSE", mse, step)

                pred = model.get_binary_graph()[:, :-1].cpu().detach().numpy()
                gt = ground_truth_causal_graph

                tp = np.sum((pred == 1) & (gt == 1))
                fp = np.sum((pred == 1) & (gt == 0))
                fn = np.sum((pred == 0) & (gt == 1))
                tn = np.sum((pred == 0) & (gt == 0))

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                writer.add_scalar("graph_eval/precision", precision, step)
                writer.add_scalar("graph_eval/recall", recall, step)
                writer.add_scalar("graph_eval/f1", f1, step)
                writer.add_scalar("graph_eval/accuracy", accuracy, step)

                metrics["graph"] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float(accuracy),
                }
                logger.info(
                    "Step %d loss=%.4f precision=%s recall=%s f1=%s accuracy=%s edges=%s",
                    step,
                    loss,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    pred.sum(),
                )

            elif len(test_buffer) > cfg.train.test_batch_size:
                s_b, a_b, s_1b = test_buffer.sample(cfg.train.test_batch_size)
                s_b, a_b, s_1b = (
                    s_b.float().to(cfg.device),
                    a_b.float().reshape(-1, action_dim).to(cfg.device),
                    s_1b.float().to(cfg.device),
                )
                mse = model.evaluate_predictions(s_b, a_b, s_1b)
                metrics["prediction_mse"] = mse
                logger.info("Next-step prediction MSE: %s", mse)
                writer.add_scalar("Predictions/MSE", mse, step)

    for episode, reward in enumerate(episode_rewards):
        writer.add_scalar("policy_stat/episode_reward", reward, episode)

    if metrics["prediction_mse"] is None and test_buffer:
        s_b, a_b, s_1b = test_buffer.sample(min(len(test_buffer), cfg.train.test_batch_size))
        s_b, a_b, s_1b = (
            s_b.float().to(cfg.device),
            a_b.float().reshape(-1, action_dim).to(cfg.device),
            s_1b.float().to(cfg.device),
        )
        metrics["prediction_mse"] = model.evaluate_predictions(s_b, a_b, s_1b)

    writer.close()

    model.save_model(filepath=checkpoint_path)
    write_metrics(run_dir, metrics)
    logger.info("Training run saved to %s", run_dir)
    return run_dir
