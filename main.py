import torch
import numpy as np
from torch.utils.data import Dataset
from Environment import get_env
import random
from Policies.RandomPolicy import RandomPolicy
from torch.utils.tensorboard import SummaryWriter
import os
from Models import get_model
from config import DEFAULT_CONFIG, ExperimentConfig
from datetime import datetime


def state_to_tensor(state_dict):
    kpi, param = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            param.append(state_dict[key])
        elif "kpi" in key:
            kpi.append(state_dict[key])
    return torch.from_numpy(np.concatenate(param + kpi)).float()


class ReplayBufferDataset(Dataset):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, s_next = self.data[idx]
        return s, a, s_next

    def add(self, s, a, s_next):
        self.data.append((s, a, s_next))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.data), size=batch_size)
        s, a, s_next = zip(*(self.data[i] for i in idx))
        return torch.stack(s), torch.tensor(a), torch.stack(s_next)


def main(cfg: ExperimentConfig = DEFAULT_CONFIG, resume=False):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = get_env(cfg)

    ground_truth_causal_graph = env.true_adj_matrix
    model_label = "CMI" if cfg.model_kind == "cdl" else "MLP"
    result_dir = f"rslts/{model_label}/{cfg.environment}-{datetime.now():%Y%m%d_%H%M%S}/"

    writer = SummaryWriter(os.path.join(result_dir, "tensorboard"))

    state_dim = env.get_state_dim()
    action_dim = env.action_dim

    random_policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    model = get_model(cfg, env)

    model_load_name = f"{model_label}-{cfg.environment}_model.pt"
    if resume:
        if os.path.exists(model_load_name):
            model.load_model(model_load_name)
            print(f"[main] Resumed from checkpoint: {model_load_name}")
        else:
            print(f"[main] No checkpoint at {model_load_name}; training from scratch.")

    train_buffer = ReplayBufferDataset(state_dim, action_dim)
    test_buffer = ReplayBufferDataset(state_dim, action_dim)
    obs = env.reset()
    loss = 0.0
    episode_reward = 0
    episode_rewards = []
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
            print(f"Episode {len(episode_rewards)} Reward : ", episode_rewards[-1])

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
                    mse = model.evaluatePredictions(s_b, a_b, s_1b)
                    print("Next Step Prediction MSE: ", mse)
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

                print(f"\nStep {step} | Loss: {loss:.4f} | Policy: random")
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1:", f1)
                print("Accuracy:", accuracy)
                print("N:", pred.sum())

            elif len(test_buffer) > cfg.train.test_batch_size:
                s_b, a_b, s_1b = test_buffer.sample(cfg.train.test_batch_size)
                s_b, a_b, s_1b = (
                    s_b.float().to(cfg.device),
                    a_b.float().reshape(-1, action_dim).to(cfg.device),
                    s_1b.float().to(cfg.device),
                )
                mse = model.evaluatePredictions(s_b, a_b, s_1b)
                print("Next Step Prediction MSE: ", mse)
                writer.add_scalar("Predictions/MSE", mse, step)

    for episode, reward in enumerate(episode_rewards):
        writer.add_scalar("policy_stat/episode_reward", reward, episode)

    writer.close()

    model.save_model(filepath=model_load_name)


if __name__ == "__main__":
    main()
