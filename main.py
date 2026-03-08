import torch
import numpy as np
from torch.utils.data import Dataset
from Environment.QACM import ORANEnvironment2
from CDL_ammended import CDL
from Policies.RandomPolicy import RandomPolicy


# ---- CONSTANTS ----
TOTAL_STEPS = 20000
COLLECT_STEPS = True
INFERENCE_GRADIENT_STEPS = 1
TRAIN_PROP = 1.0
INIT_STEPS = 3000
BATCH_SIZE = 128
PLOT_FREQ = 5000
SAVING_FREQ = 500
EVAL_STEPS = 10
CMI_THRESHOLD = 0.2
EVAL_TAU = 0.99
GRAD_CLIP  = 20001
GENERATIVE_FC_DIMS = [64, 64]
FEATURE_FC_DIMS = [64, 64]



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- STATE PROCESSING ----
def state_to_tensor(state_dict):
    kpi, param = [], []

    for key in sorted(state_dict.keys()):
        if "param" in key:
            param.append(state_dict[key])
        elif "kpi" in key:
            kpi.append(state_dict[key])

    arr = np.concatenate(param + kpi)

    return torch.from_numpy(arr).float()


# ---- REPLAY BUFFER ----
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

        s = torch.stack(s)
        s_next = torch.stack(s_next)
        a = torch.tensor(a)

        return s, a, s_next


# ---- MAIN TRAINING LOOP ----
def main():

    env = ORANEnvironment2()
    ground_truth_causal_graph = env.true_adj_matrix

    policy = RandomPolicy(
        action_dim=env.action_dim,
        num_bins=env.num_bins,
        num_params=env.num_params,
        num_kpis=env.num_kpis
    )

    state_dim = env.num_params + env.num_kpis
    action_dim = env.action_dim

    cdl = CDL(
        state_dim=state_dim,
        action_dim=action_dim,
        device=DEVICE,
        cmi_threshold=CMI_THRESHOLD,
        eval_tau=EVAL_TAU,
        grad_clip=GRAD_CLIP,
        generative_fc_dims=GENERATIVE_FC_DIMS,
        feature_fc_dims=FEATURE_FC_DIMS,
        lr=1e-3

    )

    buffer = ReplayBufferDataset(state_dim, action_dim)

    obs = env.reset()

    for step in range(TOTAL_STEPS):

        # ---- ACT ----
        action = policy.act()

        next_obs, reward, done, info = env.step(action)

        # ---- STORE TRANSITION ----
        s_tensor = state_to_tensor(obs)
        s_next_tensor = state_to_tensor(next_obs)

        buffer.add(s_tensor, action, s_next_tensor)

        obs = next_obs

        if done:
            obs = env.reset()

        # ---- COLLECT INITIAL DATA ----
        if step < INIT_STEPS:
            continue

        # ---- TRAIN CAUSAL MODEL ----
        for i in range(INFERENCE_GRADIENT_STEPS):

            s, a, s_next = buffer.sample(BATCH_SIZE)

            s = s.to(torch.float32).to(DEVICE)
            s_next = s_next.to(torch.float32).to(DEVICE)
            # a = a.float().to(DEVICE)
            a = a.float().reshape(-1, 1).to(DEVICE)

            s_pair = torch.stack([s, s_next], dim=1).to(DEVICE)

            loss = cdl.train_step(s_pair, a)

        # ---- EVALUATE CAUSAL GRAPH ----
        if step % (EVAL_STEPS * INFERENCE_GRADIENT_STEPS) == 0:

            s, a, s_next = buffer.sample(BATCH_SIZE)

            s = s.to(torch.float32).to(DEVICE)
            s_next = s_next.to(torch.float32).to(DEVICE)
            a = a.float().reshape(-1,1).to(DEVICE)

            s_pair = torch.stack([s, s_next], dim=1).to(DEVICE)

            cdl.update_mask(s_pair, a)

        if step % PLOT_FREQ == 0:
            pred = cdl.get_binary_graph()[:,:-1].cpu().detach().numpy()
            gt = ground_truth_causal_graph

            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            tn = np.sum((pred == 0) & (gt == 0))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            print("Precision:", precision)
            print("Recall:", recall)
            print("F1:", f1)
            print("Accuracy:", accuracy)
            print("N:", pred.sum())

        if step % PLOT_FREQ == 0:
            # print("Predicted Causal Graph:",pred) 
            print(cdl.get_causal_graph()[:,:-1].cpu().detach().numpy(), "\n\n\n")   




if __name__ == "__main__":
    main()