"""
intervention.py  —  Active intervention strategy for causal graph discovery.

The core problem with random exploration:
    - All 7 params change simultaneously every step
    - When K1 changes, you can't tell if P1 or P2 caused it (confounding)
    - You waste steps on params whose causal status is already clear

Strategy: Uncertainty-driven single-param interventions
    1. Look at the CMI matrix — find the param whose causal relationships
       are most ambiguous (CMI values closest to the threshold)
    2. Freeze all other params at their current values
    3. Sweep ONLY that param across its full range
    4. This gives a clean, isolated signal for that param's causal effects

Three intervention modes (automatically selected each step):
    - ISOLATE  : vary one ambiguous param, freeze all others
    - SWEEP    : push the target param to extremes (min/max) for max signal
    - RANDOM   : fallback once graph is confident (epsilon-greedy)
"""

import random
import numpy as np
import torch
from typing import List, Tuple, Optional


class InterventionPolicy:
    """
    Decides which params to intervene on and how, based on current
    CMI uncertainty. Replaces the random action_fn in environment.py.

    Usage:
        policy = InterventionPolicy(param_thresholds, cmi_threshold=0.01)

        # each env step:
        action = policy.select_action(current_params, cdl.get_cmi_matrix())
        for i, param in enumerate(params):
            param.set_param(action[i])
    """

    def __init__(
        self,
        param_thresholds: List[Tuple[float, float]],
        cmi_threshold:    float = 0.01,
        epsilon:          float = 0.1,   # prob of random fallback
        n_params:         int   = 7,
        n_kpis:           int   = 4,
    ):
        self.param_thresholds = param_thresholds   # [(low, high)] * n_params
        self.cmi_threshold    = cmi_threshold
        self.epsilon          = epsilon
        self.n_params         = n_params
        self.n_kpis           = n_kpis

        # Tracks how many steps each param has been the focus
        self.focus_counts     = np.zeros(n_params)
        # Current frozen baseline values for non-focus params
        self.frozen_values    = [
            (low + high) / 2.0 for low, high in param_thresholds
        ]
        self.step             = 0

    def _param_uncertainty(self, cmi_matrix: torch.Tensor) -> np.ndarray:
        """
        For each param i, compute its uncertainty score = how ambiguous
        its causal relationships with KPIs are.

        Ambiguity = distance from threshold, inverted.
        CMI values near the threshold are most uncertain.
        CMI values far above OR below threshold are already decided.

        Returns: (n_params,) uncertainty scores, higher = more ambiguous
        """
        if isinstance(cmi_matrix, torch.Tensor):
            cmi = cmi_matrix.cpu().numpy()
        else:
            cmi = cmi_matrix

        # Only look at param -> KPI slice: rows 0..6, cols 7..10
        param_kpi_cmi = cmi[:self.n_params, self.n_params:]  # (7, 4)

        # Distance from threshold — close to threshold = high uncertainty
        dist_from_threshold = np.abs(param_kpi_cmi - self.cmi_threshold)

        # Uncertainty per param = mean ambiguity across all KPIs
        # Invert so that smallest distance = highest uncertainty
        uncertainty = 1.0 / (dist_from_threshold.mean(axis=1) + 1e-6)

        return uncertainty  # (n_params,)

    def _select_focus_param(self, cmi_matrix: torch.Tensor) -> int:
        """
        Pick the param to intervene on this step.

        Combines:
          - Uncertainty score (CMI closest to threshold)
          - Under-explored bonus (params that haven't been focused on recently)
        """
        uncertainty  = self._param_uncertainty(cmi_matrix)

        # Normalize focus counts — params focused on less get a bonus
        focus_norm   = self.focus_counts / (self.focus_counts.sum() + 1e-6)
        explore_bonus = 1.0 - focus_norm   # higher = less explored

        # Combined score
        score = uncertainty * 0.7 + explore_bonus * 0.3

        return int(np.argmax(score))

    def select_action(
        self,
        current_params: List[float],
        cmi_matrix:     Optional[torch.Tensor] = None,
    ) -> List[float]:
        """
        Returns a list of new param values [P1..P7].

        Intervention modes:
          - If cmi_matrix is None or epsilon triggers: full random (warmup)
          - ISOLATE mode: vary focus param, freeze others at midpoint
          - SWEEP mode (every 5 steps): push focus param to extreme values
            alternating min/max for maximum causal signal

        Args:
            current_params : current [P1..P7] float values
            cmi_matrix     : (state_dim, state_dim) tensor from cdl.get_cmi_matrix()
        Returns:
            new_params : list of 7 float values
        """
        self.step += 1

        # Warmup / epsilon-greedy fallback → pure random
        if cmi_matrix is None or random.random() < self.epsilon:
            return [
                random.uniform(low, high)
                for low, high in self.param_thresholds
            ]

        # Pick the most uncertain param to focus on
        focus = self._select_focus_param(cmi_matrix)
        self.focus_counts[focus] += 1

        low, high = self.param_thresholds[focus]

        # SWEEP mode every 5 steps: alternate between min and max
        # This maximally separates cause from no-cause for the focus param
        if self.step % 5 == 0:
            focus_value = low if (self.step // 5) % 2 == 0 else high
        else:
            # ISOLATE mode: random within full range
            focus_value = random.uniform(low, high)

        # All other params frozen at their current values
        # (freezing removes confounding — only focus param changes)
        new_params = list(current_params)
        new_params[focus] = focus_value

        return new_params

    def update_frozen_values(self, current_params: List[float]):
        """Call after each step to keep frozen baseline up to date."""
        self.frozen_values = list(current_params)

    def get_focus_distribution(self) -> dict:
        """Returns how often each param was the focus — useful for diagnostics."""
        total = self.focus_counts.sum()
        if total == 0:
            return {f"P{i+1}": 0.0 for i in range(self.n_params)}
        return {
            f"P{i+1}": float(self.focus_counts[i] / total)
            for i in range(self.n_params)
        }