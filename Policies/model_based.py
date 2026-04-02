import numpy as np
import torch
from typing import Callable, Any


class ModelBasedPolicy:
    """
    A model-based policy using Cross-Entropy Method (CEM) for action selection.
    
    This policy leverages a learned dynamics model (cdl) to simulate trajectories
    and optimize actions over a planning horizon via CEM. It samples candidate
    action sequences, evaluates them using a reward function, and refines the
    distribution iteratively.
    """

    STD_CLAMP_MIN = 1e-3  # Minimum standard deviation to prevent collapse

    def __init__(
        self,
        cdl: Any,  # Dynamics model (e.g., CDL instance)
        env: Any,  # Environment providing action space and dimensions
        xapps: Any = None,  # xApps for scoring
        weights_per_xapps: list = None,  # Weights for each xapp
        scaling_term: float = 1.0,  # Scaling term (tau)
        n_horizon: int = 1,
        n_candidate: int = 500,
        n_top: int = 50,
        n_iter: int = 5,
    ):
        """
        Initialize the ModelBasedPolicy.
        
        Args:
            cdl: Learned dynamics model for state prediction.
            env: Environment object with num_bins, num_params, paramThresholds, etc.
            xapps: List of xapps for scoring (same as conflict algorithms).
            weights_per_xapps: Weight for each xapp.
            scaling_term: Scaling factor (tau) for cost computation.
            n_horizon: Planning horizon (number of steps to simulate).
            n_candidate: Number of candidate action sequences per CEM iteration.
            n_top: Number of top-performing candidates to retain as elites.
            n_iter: Number of CEM iterations.
        """
        assert n_top <= n_candidate, "n_top must be <= n_candidate"
        assert n_horizon > 0, "n_horizon must be positive"
        
        self.cdl = cdl
        self.env = env
        self.xapps = xapps if xapps is not None else env.xapps
        self.weights_per_xapps = weights_per_xapps if weights_per_xapps is not None else [1.0] * len(self.xapps)
        self.scaling_term = scaling_term
        self.n_horizon = n_horizon
        self.n_candidate = n_candidate
        self.n_top = n_top
        self.n_iter = n_iter
        self.device = cdl.device
        self.num_bins = env.num_bins
        self.num_params = env.num_params
        self.kpi_start = env.num_params
        self.state_dim = cdl.state_dim
        self.action_dim = cdl.action_dim
        
        # Calculate action bounds: [param_id_max, bin_id_max, index_max]
        self.param_thresholds = env.paramThresholds
        self.bin_lengths = [int((t[1] - t[0]) // self.num_bins) for t in self.param_thresholds]
        self.max_bin_length = max(self.bin_lengths)
        self.action_space = [self.num_params - 1, self.num_bins - 1, self.max_bin_length - 1]

    def act(self, obs_tensor: torch.Tensor) -> np.ndarray:
        """
        Select an action given the current observation.
        
        Args:
            obs_tensor: Current state observation as a tensor.
        
        Returns:
            Action as a numpy array of shape (action_dim,).
        """
        state = obs_tensor.float().to(self.device)
        
        self.cdl.models.eval()
        with torch.no_grad():
            action = self._cem(state)
        self.cdl.models.train()
        
        return action

    def _cem(self, initial_state: torch.Tensor) -> np.ndarray:
        """
        Perform Cross-Entropy Method to optimize action sequence.
        
        Args:
            initial_state: Initial state tensor of shape (state_dim,).
        
        Returns:
            Best action as numpy array of shape (action_dim,).
        """
        action_bounds = torch.tensor(self.action_space, dtype=torch.float32, device=self.device)  # (action_dim,)
        
        mean = action_bounds / 2.0  # Initialize mean at midpoint
        mean = mean.unsqueeze(0).expand(self.n_horizon, -1).clone()  # (n_horizon, action_dim)
        std_dev = mean.clone()  # (n_horizon, action_dim)
        
        for _ in range(self.n_iter):
            noise = torch.randn(self.n_candidate, self.n_horizon, self.action_dim, device=self.device)
            action_candidates = (mean + std_dev * noise).round().long()  # (n_candidate, n_horizon, action_dim)
            
            # Clamp each action dimension independently to valid bounds
            action_candidates = torch.stack([
                action_candidates[..., i].clamp(0, self.action_space[i])
                for i in range(self.action_dim)
            ], dim=-1)  # (n_candidate, n_horizon, action_dim)
            
            predicted_kpis = self._rollout(initial_state, action_candidates)  # (n_candidate, n_horizon, n_kpis)
            rewards = self._score(predicted_kpis)  # (n_candidate,) - higher is better
            
            top_indices = torch.argsort(rewards, descending=True)[:self.n_top]
            elites = action_candidates[top_indices].float()  # (n_top, n_horizon, action_dim)
            mean = elites.mean(dim=0)  # (n_horizon, action_dim)
            std_dev = elites.std(dim=0).clamp(min=self.STD_CLAMP_MIN)  # (n_horizon, action_dim)
        
        # Select the first action from the best sequence (for multi-step, this is the immediate action)
        best_action = mean[0].round().long()
        best_action = torch.stack([
            best_action[i].clamp(0, self.action_space[i])
            for i in range(self.action_dim)
        ])  # (action_dim,)
        return best_action.cpu().numpy()

    def _rollout(self, initial_state: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        """
        Simulate trajectories using the dynamics model.
        
        Args:
            initial_state: Initial state tensor of shape (state_dim,).
            action_sequences: Action sequences of shape (n_candidate, n_horizon, action_dim).
        
        Returns:
            Predicted KPIs over horizon: (n_candidate, n_horizon, n_kpis).
        """
        num_candidates = self.n_candidate
        state = initial_state.unsqueeze(0).expand(num_candidates, -1).clone().float()  # (n_candidate, state_dim)
        predicted_kpis_list = []
        
        for step in range(self.n_horizon):
            action_batch = action_sequences[:, step, :].long()  # (n_candidate, action_dim)
            
            # Apply actions to update parameters in state
            for i in range(num_candidates):
                param_id, bin_id, index = action_batch[i, 0].item(), action_batch[i, 1].item(), action_batch[i, 2].item()
                
                # Convert action to parameter value
                low, high = self.param_thresholds[param_id]
                base = low + (high - low) * (bin_id / (self.num_bins - 1))
                new_param_value = base + index
                new_param_value = np.clip(new_param_value, low, high)
                
                # Update parameter in state
                state[i, param_id] = new_param_value
            
            # Predict next KPIs from updated state
            kpi_distributions = self.cdl.predictNextState(state, action_batch.float())
            next_kpis = kpi_distributions.sample()  # (n_candidate, n_kpis)
            
            predicted_kpis_list.append(next_kpis)
            
            # Update state with new KPIs
            state[:, self.kpi_start:] = next_kpis
        
        return torch.stack(predicted_kpis_list, dim=1)  # (n_candidate, n_horizon, n_kpis)

    def _score(self, predicted_kpis: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for candidate action sequences using QACM-style scoring.
        
        Args:
            predicted_kpis: Predicted KPIs of shape (n_candidate, n_horizon, n_kpis).
        
        Returns:
            Rewards per candidate: (n_candidate,). Higher is better (negated cost).
        """
        n_candidate = predicted_kpis.shape[0]
        n_horizon = predicted_kpis.shape[1]
        kpis_np = predicted_kpis.cpu().detach().numpy()
        
        costs = np.zeros(n_candidate)
        
        for i in range(n_candidate):
            total_cost = 0.0
            for step in range(n_horizon):
                kpis = kpis_np[i, step]
                cost_vec = np.zeros(len(self.xapps))
                sat_vec = np.zeros(len(self.xapps))
                
                for j, xapp in enumerate(self.xapps):
                    u = xapp.compute_utility(kpis)
                    d, s = self._weighted_distance(xapp, u, j)
                    cost_vec[j] = self.weights_per_xapps[j] * d * self.scaling_term
                    sat_vec[j] = s
                
                f_cost = cost_vec.sum() - (sat_vec.sum()) ** 2
                total_cost += f_cost
            
            costs[i] = total_cost
        
        # Return negative cost (higher is better for maximization)
        return torch.tensor(-costs, dtype=torch.float32, device=self.device)
    
    def _weighted_distance(self, xapp, utility, xapp_idx):
        """
        Compute weighted distance for an xapp (same as QACM).
        
        Args:
            xapp: xApp object.
            utility: Computed utility value (z-score).
            xapp_idx: Index of xapp in the list.
        
        Returns:
            (distance, satisfaction) tuple.
        """
        mean, std = self.env.kpis[xapp_idx].mean, self.env.kpis[xapp_idx].std
        norm_threshold = (xapp.threshold - mean) / std
        
        if xapp.direction == 0:  # maximise
            if utility < norm_threshold:
                return norm_threshold - utility, 0
            return 0.0, 1
        else:  # minimise
            if utility > norm_threshold:
                return utility - norm_threshold, 0
            return 0.0, 1