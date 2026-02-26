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
from torch import Tensor
from typing import List, Tuple, Optional
from Environment import Param,KPI,XApp
from Parameters import INITIAL_COLLECTION_STEPS

class InterventionPolicy:
    def __init__(
        self,
        xapps:          List[XApp],
        params:          List[Param],
        kpis:            List[KPI],
        cmi_threshold:    float = 0.01,
        epsilon:          float = 0.1,
        cmi_matrix:      Tensor=None,
    ):
        self.xapps         = xapps
        self.params        = params
        self.kpis          = kpis
        self.n_params      = len(params)
        self.n_kpis        = len(kpis)
        self.n_xapps       = len(xapps)
        self.cmi_threshold = cmi_threshold
        self.epsilon       = epsilon
        self.param_thresholds = [(p.get_threshold()[0],p.get_threshold()[1]) for p in self.params]   
        self.cmi_matrix     = cmi_matrix

        self.focus_counts     = np.zeros(self.n_params)
        self.frozen_values    = [
            (low + high) / 2.0 for low, high in self.param_thresholds
        ]
        self.step             = 0

    def _param_uncertainty(self,cmi_matrix:np.ndarray) -> np.ndarray:

        param_kpi_cmi = cmi_matrix[:self.n_params, self.n_params:]  

        dist_from_threshold = np.abs(param_kpi_cmi - self.cmi_threshold)

        #less distance ,higher uncertainity
        uncertainty = 1.0 / (dist_from_threshold.mean(axis=1) + 1e-6) 

        return uncertainty  

    def _select_focus_param(self,cmi_matrix:np.ndarray) -> int:

        uncertainty  = self._param_uncertainty(cmi_matrix)

        focus_norm   = self.focus_counts / (self.focus_counts.sum() + 1e-6)

        #focus low means explored very few times,so 1-focus prioritizes those that have not been explored 
        explore_bonus = 1.0 - focus_norm   

        # 70% weight given to uncertainity and 30% weight given to unexplored params
        score = uncertainty * 0.7 + explore_bonus * 0.3

        return int(np.argmax(score))

    def select_action(
        self,
        cmi_matrix:np.ndarray
    ) -> Tuple[float,float]:
        
        if cmi_matrix is None or random.random() < self.epsilon:
            return -1,-1

        focus = self._select_focus_param(cmi_matrix)
        self.focus_counts[focus] += 1

        low, high = self.param_thresholds[focus]


        if self.step % 5 == 0: #SWEEP Mode
            focus_value = low if (self.step // 5) % 2 == 0 else high
        else:
            # ISOLATE mode
            focus_value = random.uniform(low, high)

        return focus,focus_value

    def update_frozen_values(self, current_params: List[float]):
        self.frozen_values = list(current_params)

    def get_focus_distribution(self) -> dict:
        total = self.focus_counts.sum()
        if total == 0:
            return {f"P{i+1}": 0.0 for i in range(self.n_params)}
        return {
            f"P{i+1}": float(self.focus_counts[i] / total)
            for i in range(self.n_params)
        }
    
    def randomExplore(self):
        for xapp in self.xapps:
            xapp.action()
    
    def act(self):
        self.step = self.step + 1
        if self.step <= INITIAL_COLLECTION_STEPS:
            self.randomExplore()
            return

        cmi_matrix = self.cmi_matrix.detach().cpu().numpy().copy()
        focus,focus_value = self.select_action(cmi_matrix=cmi_matrix)
        
        if focus == -1:
            self.randomExplore()
            return

        param=self.params[focus]
        param.set_param(focus_value)


