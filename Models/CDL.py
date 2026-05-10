import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import networkx as nx
import matplotlib.pyplot as plt


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StatePredictor(nn.Module):

    def __init__(self, state_dim, action_dim, feature_dim, pred_hidden):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, feature_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, feature_dim, [])
        self.predictor = MLP(feature_dim, 2, pred_hidden)

    def forward(self, s, a, mask=None):
        """
        s:    (bs, state_dim)
        a:    (bs, action_dim)
        mask: (bs, state_dim+1) bool
        returns: mu (bs, 1), std (bs, 1)
        """
        bs = s.shape[0]
        fd = self.state_dim

        feats = []
        for i in range(fd):
            feats.append(self.state_feature_extractors[i](s[:, i:i+1]))

        feats.append(self.action_feature_extractor(a))
        feats = torch.stack(feats, dim=1)                         # (bs, fd+1, feature_dim)

        if mask is not None:
            feats = feats.masked_fill(mask.unsqueeze(-1), float('-inf'))

        h, _ = feats.max(dim=1)                                   # (bs, feature_dim)

        out     = self.predictor(h)
        mu      = out[:, 0:1]
        log_std = out[:, 1:2]
        std     = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4

        return mu, std

class CDL:

    def __init__(
        self,
        state_dim,
        action_dim,
        kpi_start,
        feature_fc_dims=(64,),
        generative_fc_dims=(64, 32),
        lr=3e-4,
        cmi_threshold=0.2,
        eval_tau=0.99,
        eval_steps=10,
        grad_clip=10.0,
        device=None,
        node_names=None
    ):
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.cmi_threshold = cmi_threshold
        self.eval_tau      = eval_tau
        self.eval_steps    = eval_steps
        self.grad_clip     = grad_clip
        self.kpi_start = kpi_start
        self.node_names = node_names

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        feature_dim = feature_fc_dims[-1]

        self.models = nn.ModuleList([
            StatePredictor(state_dim, action_dim, feature_dim, list(generative_fc_dims))
            for _ in range(state_dim)
        ]).to(self.device)

        self.opt = optim.Adam(self.models.parameters(), lr=lr)

        fd = state_dim
        self.mask_CMI         = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_cmi_acc    = torch.zeros(fd, fd + 1, device=self.device)
        self._eval_step_count = 0

    def _nll(self, mu, std, target):
        return -Normal(mu, std).log_prob(target)

    def train_step(self, s_batch, a_batch):
        s_t   = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs    = s_t.shape[0]
        fd    = self.state_dim

        self.opt.zero_grad()
        total_loss = 0.0

        # a_batch[:, 0] is param_id — the param that was changed
        # bias: drop the changed param more often so model learns to predict without it
        changed_param_ids = a_batch[:, 0].long()           # (bs,) which param changed
        
        # 50% of the time drop the changed param, 50% drop random
        use_informed = torch.rand(bs, device=self.device) > 0.5
        random_drop  = torch.randint(fd + 1, (bs,), device=self.device)
        informed_drop = changed_param_ids                  # drop the changed param
        
        drop_idx = torch.where(use_informed, informed_drop, random_drop)
        mask     = F.one_hot(drop_idx, fd + 1).bool()

        for j in range(fd):
            target = s_tp1[:, j:j+1]
            model  = self.models[j]
            model.train()

            mu, std     = model(s_t, a_batch)
            full_loss   = self._nll(mu, std, target).mean()

            mu_m, std_m = model(s_t, a_batch, mask=mask)
            masked_loss = self._nll(mu_m, std_m, target).mean()

            total_loss += full_loss + masked_loss

        loss = total_loss / fd
        loss.backward()
        nn.utils.clip_grad_norm_(self.models.parameters(), self.grad_clip)
        self.opt.step()

        return loss.item()

    def update_mask(self, s_batch, a_batch):
        s_t   = s_batch[:, 0]
        s_tp1 = s_batch[:, 1]
        bs    = s_t.shape[0]
        fd    = self.state_dim

        step_cmi = torch.zeros(fd, fd + 1, device=self.device)

        with torch.no_grad():
            for j in range(fd):
                target   = s_tp1[:, j:j+1]
                self.models[j].eval()
                mu, std  = self.models[j](s_t, a_batch)
                full_nll = self._nll(mu, std, target)

                for i in range(fd + 1):
                    mask = torch.zeros(bs, fd + 1, dtype=torch.bool, device=self.device)
                    self.models[j].eval()
                    mask[:, i] = True
                    mu_m, std_m = self.models[j](s_t, a_batch, mask=mask)
                    masked_nll  = self._nll(mu_m, std_m, target)
                    step_cmi[j, i] = (masked_nll - full_nll).mean()

        self._eval_cmi_acc    += step_cmi
        self._eval_step_count += 1

        if self._eval_step_count >= self.eval_steps:
            avg_cmi       = self._eval_cmi_acc / self.eval_steps
            self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg_cmi
            self._eval_cmi_acc    = torch.zeros(fd, fd + 1, device=self.device)
            self._eval_step_count = 0

    def get_causal_graph(self):
        return self.mask_CMI

    def get_binary_graph(self):
        graph = (self.mask_CMI >= self.cmi_threshold)
        fd = graph.shape[0]
        if graph.shape[1] > fd:
            graph[:fd, :fd].fill_diagonal_(0)
        return graph

    def predictNextState(self, s, a):
        bs = s.shape[0]
        s = s.to(self.device)
        a = a.to(self.device)
        mus, stds = [], []
        with torch.no_grad():
            for j in range(self.kpi_start, self.state_dim):
                self.models[j].eval()
                graph_mask = self.get_binary_graph()[j, :].clone()  # (fd+1,)
                graph_mask[j]  = True   # always include self
                graph_mask[-1] = True   # always include action
                mask = graph_mask.unsqueeze(0).expand(bs, -1).bool().to(self.device)
                mu, std = self.models[j](s, a, ~mask)
                mus.append(mu)
                stds.append(std)
        mu  = torch.cat(mus, dim=1)
        std = torch.cat(stds, dim=1)
        return Normal(mu, std)

    def evaluatePredictions(self, s, a, s_1):
        """
        s:   (bs, state_dim)
        a:   (bs, 1)
        s_1: (bs, state_dim)
        returns: scalar MSE
        """
        dist   = self.predictNextState(s, a)
        pred   = dist.sample()                        # (bs, state_dim - kpi_start)
        target = s_1[:, self.kpi_start:]              # (bs, state_dim - kpi_start)
        return ((pred - target) ** 2).mean().item()   # scalar

        
    def save_model(self, filepath="cdl_model.pt"):
        """
        Save the model state, optimizer state, and other necessary attributes.
        """
        state = {
            'models_state_dict': [model.state_dict() for model in self.models],
            'optimizer_state_dict': self.opt.state_dict(),
            'mask_CMI': self.mask_CMI,
            'eval_cmi_acc': self._eval_cmi_acc,
            'eval_step_count': self._eval_step_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'cmi_threshold': self.cmi_threshold,
            'eval_tau': self.eval_tau,
            'eval_steps': self.eval_steps,
            'grad_clip': self.grad_clip,
            'device': self.device,
        }
        torch.save(state, filepath)

    def load_model(self, filepath="cdl_model.pt"):
        """
        Load the model state, optimizer state, and other attributes.
        """
        state = torch.load(filepath, map_location=self.device)
        for i, model in enumerate(self.models):
            model.load_state_dict(state['models_state_dict'][i])
        self.opt.load_state_dict(state['optimizer_state_dict'])
        self.mask_CMI = state['mask_CMI']
        self._eval_cmi_acc = state['eval_cmi_acc']
        self._eval_step_count = state['eval_step_count']


    def visualize_causal_graph(self, threshold=None):
        graph = self.get_binary_graph()
        fd = graph.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(fd))

        for i in range(fd):
            for j in range(fd):
                if graph[i, j]:
                    G.add_edge(j, i)

        ncp_nodes = list(range(self.kpi_start))
        kpi_nodes = list(range(self.kpi_start, fd))

        n_ncp = len(ncp_nodes)
        n_kpi = len(kpi_nodes)

        pos = {}
        for idx, node in enumerate(ncp_nodes):
            pos[node] = (idx * 2.0 / max(n_ncp - 1, 1), 1.0)
        for idx, node in enumerate(kpi_nodes):
            pos[node] = (idx * 2.0 / max(n_kpi - 1, 1), 0.0)

        labels = {i: self.node_names[i] for i in range(fd)}

        ncp_to_kpi_edges = [(u, v) for u, v in G.edges() if u in ncp_nodes and v in kpi_nodes]
        kpi_to_kpi_edges = [(u, v) for u, v in G.edges() if u in kpi_nodes and v in kpi_nodes]
        ncp_to_ncp_edges = [(u, v) for u, v in G.edges() if u in ncp_nodes and v in ncp_nodes]

        plt.figure(figsize=(12, 6))

        nx.draw_networkx_nodes(G, pos, nodelist=ncp_nodes, node_color='#AED6F1',
                            node_size=1200, edgecolors='#2E86C1', linewidths=2)
        nx.draw_networkx_nodes(G, pos, nodelist=kpi_nodes, node_color='#FADBD8',
                            node_size=1200, edgecolors='#E74C3C', linewidths=2)

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        nx.draw_networkx_edges(G, pos, edgelist=ncp_to_kpi_edges, edge_color='gray',
                            arrows=True, arrowsize=15)
        nx.draw_networkx_edges(G, pos, edgelist=kpi_to_kpi_edges, edge_color='#A569BD',
                            arrows=True, arrowsize=15, style='dashed',
                            connectionstyle='arc3,rad=0.3')
        nx.draw_networkx_edges(G, pos, edgelist=ncp_to_ncp_edges, edge_color='#2E86C1',
                            arrows=True, arrowsize=15,
                            connectionstyle='arc3,rad=0.3')

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#AED6F1',
                    markeredgecolor='#2E86C1', markersize=12, label='NCP (Control)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FADBD8',
                    markeredgecolor='#E74C3C', markersize=12, label='KPI (Metric)'),
            plt.Line2D([0], [0], color='gray', label='NCP→KPI'),
            plt.Line2D([0], [0], color='#A569BD', linestyle='dashed', label='KPI→KPI Implicit'),
            plt.Line2D([0], [0], color='#2E86C1', label='NCP→NCP'),
        ]
        plt.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=True)

        plt.title("Causal Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_cmi_heatmap(self):
        import matplotlib.pyplot as plt
        import matplotlib.widgets as widgets
        import numpy as np

        cmi = self.get_causal_graph().cpu().numpy()
        fd = cmi.shape[0]

        for i in range(fd):
            cmi[i, i] = 0.0

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.25)

        im = axes[0].imshow(cmi, cmap='hot', aspect='auto')
        axes[0].set_xticks(range(cmi.shape[1]))
        axes[0].set_yticks(range(fd))
        axes[0].set_xticklabels(self.node_names + ['action'], rotation=45, ha='right', fontsize=9)
        axes[0].set_yticklabels(self.node_names, fontsize=9)
        axes[0].set_title("CMI Heatmap (raw values)")
        axes[0].set_xlabel("Source")
        axes[0].set_ylabel("Target")
        plt.colorbar(im, ax=axes[0])

        binary = (cmi >= self.cmi_threshold).astype(float)
        im2 = axes[1].imshow(binary, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[1].set_xticks(range(cmi.shape[1]))
        axes[1].set_yticks(range(fd))
        axes[1].set_xticklabels(self.node_names + ['action'], rotation=45, ha='right', fontsize=9)
        axes[1].set_yticklabels(self.node_names, fontsize=9)
        title2 = axes[1].set_title(f"Binary Graph (threshold={self.cmi_threshold})")
        axes[1].set_xlabel("Source")
        axes[1].set_ylabel("Target")
        plt.colorbar(im2, ax=axes[1])

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        slider = widgets.Slider(
            ax_slider, 'Threshold',
            valmin=0.0, valmax=float(cmi.max()),
            valinit=self.cmi_threshold, valstep=0.01
        )

        def update(val):
            t = slider.val
            binary = (cmi >= t).astype(float)
            for i in range(fd):
                binary[i, i] = 0.0
            im2.set_data(binary)
            title2.set_text(f"Binary Graph (threshold={t:.2f})")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()