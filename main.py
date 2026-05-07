import torch
import numpy as np
from torch.utils.data import Dataset
from Environment import get_env,ORANEnvironment,ORANEnvironment2
from Policies.model_based import ModelBasedPolicy
import random
from Policies.RandomPolicy import RandomPolicy
from torch.utils.tensorboard import SummaryWriter
import os
from Parameters import *
from Models import get_model

np.random.seed(SEED)
torch.manual_seed(SEED)
IS_TRAIN = True

def state_to_tensor(state_dict):
    kpi, param = [], []
    for key in sorted(state_dict.keys()):
        if "param" in key:
            param.append(state_dict[key])
        elif "kpi" in key:
            kpi.append(state_dict[key])
    return torch.from_numpy(np.concatenate(param + kpi)).float()


def oran_reward(pred_kpis, actions):
    """
    pred_kpis: (n_candidate, n_horizon, n_kpis)
    actions:   (n_candidate, n_horizon, action_dim)
    Returns:   (n_candidate, n_horizon)

    Maximise sum of all predicted KPIs across horizon.
    Replace with your actual reward definition if needed.
    """
    return pred_kpis.sum(dim=-1)


class ReplayBufferDataset(Dataset):

    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.data       = []

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


def main():
    env = get_env()

    ground_truth_causal_graph = env.true_adj_matrix
    global USE_CMI,USE_MLP,RESULT_DIR
    if isinstance(env,ORANEnvironment2):
        RESULT_DIR += "QACM"
    elif isinstance(env,ORANEnvironment):
        RESULT_DIR += "EnvironmentII"

    writer = SummaryWriter(os.path.join(RESULT_DIR, "tensorboard"))

    state_dim  = env.get_state_dim()
    action_dim = env.action_dim

    random_policy = RandomPolicy(
        action_dim=env.action_dim,
        action_space=env.action_space
    )
    mb_policy = None   

    model = get_model(env)

    if not IS_TRAIN or IS_TEST:
        model.load_model(MODEL_LOAD_NAME)

    train_buffer = ReplayBufferDataset(state_dim, action_dim)
    test_buffer = ReplayBufferDataset(state_dim,action_dim)
    obs    = env.reset()
    loss   = 0.0
    episode_reward = 0
    episode_rewards = []
    for step in range(TOTAL_STEPS):

        if mb_policy is not None:
            s_tensor = state_to_tensor(obs).to(DEVICE)
            action   = mb_policy.act(s_tensor)
        else:
            action = random_policy.act()

        next_obs, reward, done, info = env.step(action)

        s_tensor      = state_to_tensor(obs)
        s_next_tensor = state_to_tensor(next_obs)
        train_buffer.add(s_tensor, action, s_next_tensor) if random.random()>0.2 else test_buffer.add(s_tensor,action,s_next_tensor)
        episode_reward+=reward

        obs = next_obs
        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            print(f"Episode {len(episode_rewards)} Reward : ",episode_rewards[-1])

            episode_reward=0


        if (step < INIT_STEPS) and IS_TRAIN:
            continue

        if IS_TRAIN:
            for _ in range(INFERENCE_GRADIENT_STEPS):
                s, a, s_next = train_buffer.sample(BATCH_SIZE)
                s      = s.float().to(DEVICE)
                s_next = s_next.float().to(DEVICE)
                a      = a.float().reshape(-1, action_dim).to(DEVICE)
                s_pair = torch.stack([s, s_next], dim=1).to(DEVICE)
                loss   = model.train_step(s_pair, a)

            if USE_CMI:
                # ---- UPDATE CAUSAL GRAPH ----
                if step % (EVAL_STEPS * INFERENCE_GRADIENT_STEPS) == 0:
                    s, a, s_next = train_buffer.sample(BATCH_SIZE)
                    s      = s.float().to(DEVICE)
                    s_next = s_next.float().to(DEVICE)
                    a      = a.float().reshape(-1, action_dim).to(DEVICE)
                    s_pair = torch.stack([s, s_next], dim=1).to(DEVICE)
                    model.update_mask(s_pair, a)
            


        if step == MODEL_BASED_START:
            print(f"\nStep {step}: CDL trained — switching to model-based policy")
            mb_policy = ModelBasedPolicy(
                cdl=model,
                env=env,
                reward_fn=oran_reward,
                n_horizon=N_HORIZON,
                n_candidate=N_CANDIDATE,
                n_top=N_TOP,
                n_iter=N_ITER,
            )

        if (step % PLOT_FREQ == 0):

            if USE_CMI:
                if len(test_buffer)>=TEST_BATCH_SIZE:
                    s_b,a_b,s_1b = test_buffer.sample(TEST_BATCH_SIZE)
                    s_b,a_b,s_1b = s_b.float().to(DEVICE),a_b.float().reshape(-1,action_dim).to(DEVICE),s_1b.float().to(DEVICE)
                    mse = model.evaluatePredictions(s_b,a_b,s_1b)
                    print("Next Step Prediction MSE: ",mse)
                    writer.add_scalar("Predictions/MSE",mse,step)

                pred = model.get_binary_graph()[:, :-1].cpu().detach().numpy()
                gt   = ground_truth_causal_graph

                tp = np.sum((pred == 1) & (gt == 1))
                fp = np.sum((pred == 1) & (gt == 0))
                fn = np.sum((pred == 0) & (gt == 1))
                tn = np.sum((pred == 0) & (gt == 0))

                precision = tp / (tp + fp + 1e-8)
                recall    = tp / (tp + fn + 1e-8)
                f1        = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy  = (tp + tn) / (tp + tn + fp + fn)

                mode = "model-based" if mb_policy is not None else "random"

                writer.add_scalar("graph_eval/precision", precision, step)
                writer.add_scalar("graph_eval/recall", recall, step)
                writer.add_scalar("graph_eval/f1", f1, step)
                writer.add_scalar("graph_eval/accuracy", accuracy, step)

                print(f"\nStep {step} | Loss: {loss:.4f} | Policy: {mode}")
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1:", f1)
                print("Accuracy:", accuracy)
                print("N:", pred.sum())

            elif USE_MLP and len(test_buffer)>TEST_BATCH_SIZE:
                s_b,a_b,s_1b = test_buffer.sample(TEST_BATCH_SIZE)
                s_b,a_b,s_1b = s_b.float().to(DEVICE),a_b.float().reshape(-1,action_dim).to(DEVICE),s_1b.float().to(DEVICE)
                mse = model.evaluatePredictions(s_b,a_b,s_1b)
                print("Next Step Prediction MSE: ",mse)
                writer.add_scalar("Predictions/MSE",mse,step)

    
    for episode, reward in enumerate(episode_rewards):
        writer.add_scalar("policy_stat/episode_reward", reward, episode)

    writer.close()

    if IS_TRAIN:
        model.save_model(filepath = MODEL_SAVE_NAME)
if __name__ == "__main__":
    main()