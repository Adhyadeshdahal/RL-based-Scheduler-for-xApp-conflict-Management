import numpy as np
import torch
import math
from Parameters import *


class MCTSNode:
    """A single node in the search tree, representing one (bin_id, index) action."""
    __slots__ = ("action", "parent", "children",
                 "visits", "total_cost", "untried")

    def __init__(self, action, parent, untried_actions):
        self.action      = action          # (bin_id, index) tuple or None for root
        self.parent      = parent
        self.children    = []
        self.visits      = 0
        self.total_cost  = 0.0
        self.untried     = list(untried_actions)


class ModelBasedMCTS:
    """
    Monte Carlo Tree Search planning.
    Same act() interface as QACM, ModelBasedCEM, ModelBasedMPPI.

    Builds a (bin_id, index) search tree over n_simulations rollouts,
    guided by UCB1 to balance exploration and exploitation.
    Best for smaller action spaces where revisiting nodes is worthwhile.
    """

    def __init__(
        self,
        model,
        env,
        n_simulations=N_SIMULATIONS,
        ucb_c=USB_C, 
    ):
        self.model        = model
        self.env          = env
        self.xapps        = env.xapps
        self.num_bins     = env.num_bins
        self.num_params   = env.num_params
        self.action_space = env.action_space
        self.n_simulations = n_simulations
        self.ucb_c        = ucb_c
        self.device       = model.device
        self.name = "ModelBasedMCTS"



    def act(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        s0    = current_state
        pi    = conflict_param_index
        xapps = xapps_under_conflict
        w     = weights_per_xapps
        tau   = scaling_term

        # All (bin_id, index) pairs reachable from the root
        all_actions = [
            (b, idx)
            for b   in range(self.action_space[1] + 1)
            for idx in range(self.action_space[2] + 1)
        ]

        root = MCTSNode(action=None, parent=None, untried_actions=all_actions)

        for _ in range(self.n_simulations):
            node = root

            # ── Selection: descend by UCB1 until a leaf ───────────────
            while not node.untried and node.children:
                node = self._ucb_select(node)

            # ── Expansion: try one untried action ──────────────────────
            if node.untried:
                action_2d = node.untried.pop()
                child = MCTSNode(
                    action=action_2d,
                    parent=node,
                    untried_actions=[]    # leaf — expand lazily
                )
                node.children.append(child)
                node = child

            # ── Simulation: evaluate this action with the model ────────
            if node.action is not None:
                cost = self._evaluate(node.action, pi, s0, xapps, w, tau)
            else:
                cost = 0.0   # root has no action yet

            # ── Backpropagation: update visit counts and costs ─────────
            while node is not None:
                node.visits     += 1
                node.total_cost += cost
                node = node.parent

        # Pick child of root with lowest average cost
        best_child = min(
            root.children,
            key=lambda c: c.total_cost / c.visits if c.visits > 0 else float('inf')
        )
        best_bin, best_idx = best_child.action
        return [pi, best_bin, best_idx]


    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """
        Select child minimising UCB1 adapted for costs
        (lower cost = better, so we subtract the exploration bonus).
        """
        log_parent = math.log(node.visits + 1)
        def ucb_score(child):
            if child.visits == 0:
                return float('-inf')    # always try unvisited children first
            avg_cost  = child.total_cost / child.visits
            explore   = self.ucb_c * math.sqrt(log_parent / child.visits)
            return -(avg_cost - explore)  # negate: higher score = lower cost

        return max(node.children, key=ucb_score)


    def _evaluate(self, action_2d, pi, s0, xapps, w, tau) -> float:
        bin_id, idx = action_2d
        action = [pi, bin_id, idx]
        action_tensor = torch.tensor(
            action, dtype=torch.float32
        ).unsqueeze(0).to(self.device)               # (1, 3)

        next_state_dist = self.model.predictNextState(
            s0.unsqueeze(0).float(), action_tensor
        )
        kpis = next_state_dist.mean.squeeze(0).cpu().detach().numpy()

        cost_vec = np.zeros(len(xapps))
        sat_vec  = np.zeros(len(xapps))
        for i, xapp in enumerate(xapps):
            u           = xapp.compute_utility(kpis)
            d, s        = self._weighted_distance(xapp, u)
            cost_vec[i] = w[i] * d * tau
            sat_vec[i]  = s
        return float(cost_vec.sum() - (sat_vec.sum()) ** 2)

    @staticmethod
    def _weighted_distance(xapp, utility):
        if xapp.direction == 0:
            if utility < xapp.threshold:
                return xapp.threshold - utility, 0
            return 0.0, 1
        else:
            if utility > xapp.threshold:
                return utility - xapp.threshold, 0
            return 0.0, 1