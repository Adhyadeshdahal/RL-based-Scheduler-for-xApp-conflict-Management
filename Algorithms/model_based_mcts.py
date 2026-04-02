import numpy as np
import torch
import math
from Parameters import *


class MCTSNode:
    """A single node in the search tree, representing one (bin_id, index) action."""
    __slots__ = ("action", "parent", "children",
                 "visits", "total_cost", "untried")

    def __init__(self, action, parent, untried_actions):
        self.action     = action
        self.parent     = parent
        self.children   = []
        self.visits     = 0
        self.total_cost = 0.0
        self.untried    = list(untried_actions)


class ModelBasedMCTS:
    """
    Monte Carlo Tree Search planning.
    Same act() interface as QACM, ModelBasedCEM, ModelBasedMPPI.
    """

    def __init__(
        self,
        model,
        env,
        n_simulations = N_SIMULATIONS,
        ucb_c         = USB_C,
    ):
        self.model         = model
        self.env           = env
        self.xapps         = env.xapps
        self.num_bins      = env.num_bins
        self.num_params    = env.num_params
        self.action_space  = env.action_space
        self.n_simulations = n_simulations
        self.ucb_c         = ucb_c
        self.device        = model.device
        self.name          = "ModelBasedMCTS"

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

        # FIX 3: use per-param bin-length as upper bound for index
        max_index = self.env.action_space[pi + 2]

        # FIX 5: root carries only bin_id choices; each bin_id child carries
        # index choices. This gives the tree depth=2 and proper UCB1 guidance,
        # instead of all 200 leaf actions crammed into the root's untried list.
        bin_ids = list(range(self.action_space[1] + 1))
        root    = MCTSNode(action=None, parent=None, untried_actions=bin_ids)

        for _ in range(self.n_simulations):
            node = root

            # ── Selection: descend by UCB1 until a node with untried children
            while not node.untried and node.children:
                node = self._ucb_select(node)

            # ── Expansion ────────────────────────────────────────────────
            if node.untried:
                if node.action is None:
                    # Expanding root: create a bin_id child
                    bin_id = node.untried.pop()
                    index_choices = list(range(max_index + 1))   # FIX 3
                    child = MCTSNode(
                        action=bin_id,
                        parent=node,
                        untried_actions=index_choices,
                    )
                else:
                    # Expanding a bin_id node: create a (bin_id, index) leaf
                    index = node.untried.pop()
                    child = MCTSNode(
                        action=(node.action, index),
                        parent=node,
                        untried_actions=[],
                    )
                node.children.append(child)
                node = child

            # ── Simulation: evaluate only at depth-2 leaves ───────────
            if isinstance(node.action, tuple):
                cost = self._evaluate(node.action, pi, s0, xapps, w, tau)
            else:
                # Still at a bin_id node — do a random rollout
                if node.action is not None:
                    rand_idx = np.random.randint(0, max_index + 1)   # FIX 3
                    cost     = self._evaluate((node.action, rand_idx),
                                              pi, s0, xapps, w, tau)
                else:
                    cost = 0.0

            # ── Backpropagation ────────────────────────────────────────
            while node is not None:
                node.visits     += 1
                node.total_cost += cost
                node = node.parent

        # Pick the best (bin_id, index) leaf among all depth-2 children
        best_cost  = float('inf')
        best_bin   = 0
        best_idx   = 0
        for bin_node in root.children:
            for leaf in bin_node.children:
                if leaf.visits == 0:
                    continue
                avg = leaf.total_cost / leaf.visits
                if avg < best_cost:
                    best_cost = avg
                    best_bin, best_idx = leaf.action
        return [pi, best_bin, best_idx]

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 adapted for cost minimisation."""
        log_parent = math.log(node.visits + 1)

        def ucb_score(child):
            if child.visits == 0:
                # FIX 5b: large finite value so unvisited nodes are preferred
                # but don't dominate once the tree has grown
                return 1e9
            avg_cost = child.total_cost / child.visits
            explore  = self.ucb_c * math.sqrt(log_parent / child.visits)
            return -(avg_cost - explore)  # higher = better (lower cost)

        return max(node.children, key=ucb_score)

    def _evaluate(self, action_2d, pi, s0, xapps, w, tau) -> float:
        bin_id, idx   = action_2d
        action_tensor = torch.tensor(
            [pi, bin_id, idx], dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        next_state_dist = self.model.predictNextState(
            s0.unsqueeze(0).float(), action_tensor
        )
        # Model returns KPI portion only
        next_state = next_state_dist.sample().squeeze(0)
        kpis       = next_state.cpu().detach().numpy()

        cost_vec = np.zeros(len(xapps))
        sat_vec  = np.zeros(len(xapps))
        for i, xapp in enumerate(xapps):
            u           = xapp.compute_utility(kpis)
            d, s        = self._weighted_distance(xapp, u, i)
            cost_vec[i] = w[i] * d * tau
            sat_vec[i]  = s
        f_cost = cost_vec.sum() - (sat_vec.sum()) ** 2
        return float(f_cost)

    def _weighted_distance(self, xapp, utility, xapp_idx):
        # FIX 2: normalise raw KPI threshold into z-score space
        mean, std      = self.env.kpis[xapp_idx].mean, self.env.kpis[xapp_idx].std
        norm_threshold = (xapp.threshold - mean) / std

        if xapp.direction == 0:
            if utility < norm_threshold:
                return norm_threshold - utility, 0
            return 0.0, 1
        else:
            if utility > norm_threshold:
                return utility - norm_threshold, 0
            return 0.0, 1