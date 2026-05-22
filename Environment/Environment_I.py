import numpy as np
import gym
from typing import List, Callable, Tuple
from math import exp
from Parameters import ENVIRONMENT_I_MEAN_STDS,ENVIRONMENT_I_PARAM_RANGES
KPI_THRESHOLDS = [0.2, 0.6, 0.5, 0.5] 
MEAN_STD_KPIS  = ENVIRONMENT_I_MEAN_STDS



class XApp:
    def __init__(self, threshold, utility_fn, name, params, direction=0, mean_std=None):
        self.thresholds      = threshold
        self.threshold       = threshold
        self.compute_utility = utility_fn
        self.name            = name
        self.params          = params
        self.direction       = direction
        if mean_std:
            self.mean = mean_std[0]
            self.std  = mean_std[1]


def compute_utility_value(value, mean_std):
    mean, std = mean_std
    return (value - mean) / std

def compute_A1_utility(kpis):
    return compute_utility_value(kpis[0], MEAN_STD_KPIS[0])

def compute_A2_utility(kpis):
    return compute_utility_value(kpis[1], MEAN_STD_KPIS[1])

def compute_A3_utility(kpis):
    return compute_utility_value(kpis[2], MEAN_STD_KPIS[2])

def compute_A4_utility(kpis):
    return compute_utility_value(kpis[3], MEAN_STD_KPIS[3])


def RewardFn(kpis, action):
    return sum(kpis)


def set_param_default(threshold, value, params):
    low, high = threshold
    return np.clip(value, low, high)


class Param:
    def __init__(self, threshold: Tuple[float, float], id: int, set_param_fn=None):
        self.threshold    = threshold
        self.id           = id
        self.value        = np.random.uniform(*threshold)
        self.set_param_fn = set_param_fn if set_param_fn is not None else set_param_default

    def get_param(self):
        return self.value

    def set_param(self, value, params=None):
        self.value = self.set_param_fn(self.threshold, value, params)

    def get_threshold(self):
        return self.threshold


class KPI:
    def __init__(self, name: str, kpi_threshold: float, direction: int,
                 mean: float, std: float, updatefn: Callable):
        self.name          = name
        self.updatefn      = updatefn
        self.value         = 0.0
        self.kpi_threshold = kpi_threshold
        self.direction     = direction
        self.mean          = mean
        self.std           = std

    def update_kpi(self, prev_params: List[float], prev_kpis: List[float]):
        self.value = self.updatefn(prev_params, prev_kpis)
        return self.value

    def compute_utility_value(self):
        return (self.value - self.mean) / self.std

    def compute_utility_threshold(self):
        return (self.kpi_threshold - self.mean) / self.std

    def get_kpi(self):
        return self.value


# ── KPI equations from the paper ────────────────────────────────────────────
# K1 = 0.5 · exp(-(P1+50)² / (2·P2)²)
# K2 = exp(-(P1-50)²       / (2·P3)²)
# K3 = exp(-(P4+K1)²       / (2·P5)²)
# K4 = exp(-(P7+K2)²       / (2·P6)²)
#
# Conflict graph (from Fig. 3 / image):
#   A1 → P1, P2        A1 monitors K1
#   A2 → P1, P3        A2 monitors K2   (direct conflict with A1 on P1)
#   A3 → P4, P5        A3 monitors K3
#   A4 → P6, P7        A4 monitors K4
#
# Indirect conflict : P1 → K1, P1 → K2  (same param, two KPIs)
# Indirect conflict : P1 → K2, P3 → K2  (two params → same KPI K2)
# Implicit conflict : K2 feeds into K4 update, linking A2↔A4
# Implicit conflict : K1 feeds into K3 update, linking A1↔A3
# ────────────────────────────────────────────────────────────────────────────

def update_Kpi1(prev_params, prev_kpis):
    P1 = prev_params[0]
    P2 = max(prev_params[1], 1e-3)
    return 0.5 * exp(-(P1 + 50) ** 2 / (2 * P2) ** 2)

def update_Kpi2(prev_params, prev_kpis):
    P1 = prev_params[0]
    P3 = max(prev_params[2], 1e-3)
    return exp(-(P1 - 50) ** 2 / (2 * P3) ** 2)

def update_Kpi3(prev_params, prev_kpis):
    P4 = prev_params[3]
    P5 = max(prev_params[4], 1e-3)
    K1 = prev_kpis[0]
    return exp(-(P4 + K1) ** 2 / (2 * P5) ** 2)

def update_Kpi4(prev_params, prev_kpis):
    P7 = prev_params[6]
    P6 = max(prev_params[5], 1e-3)
    K2 = prev_kpis[1]
    return exp(-(P7 + K2) ** 2 / (2 * P6) ** 2)


class ORANEnvironment(gym.Env):
    """
    Conflict environment matching the model in:
      Banerjee et al., "Toward Control and Coordination in Cognitive
      Autonomous Networks," IEEE TNSM 2021.
    and used as the evaluation environment in:
      Zolghadr et al., "Learning and Reconstructing Conflicts in O-RAN:
      A Graph Neural Network Approach," arXiv:2412.14119.

    Conflict graph (Fig. 3):
      Direct   : A1 & A2 both subscribe to P1
      Indirect : P1,P3 → K2  /  P1 → K1,K2
      Implicit : K1 → K3 update (A1↔A3) / K2 → K4 update (A2↔A4)

    Parameters     : P1,P2 ∈ [0,300]   P3–P7 ∈ [0,3]   (Table I)
    KPI equations  : as listed in Section V-A of the paper
    """

    def __init__(self, num_bins=10, max_steps=50):
        super().__init__()

        self.paramThresholds = ENVIRONMENT_I_PARAM_RANGES
        setParamFns = [set_param_default] * 7

        self.params = [
            Param(self.paramThresholds[i], i, setParamFns[i])
            for i in range(7)
        ]

        kpi_names    = ["kpi1", "kpi2", "kpi3", "kpi4"]
        direction    = [0, 0, 0, 0]
        updateKPIFns = [update_Kpi1, update_Kpi2, update_Kpi3, update_Kpi4]

        self.kpis = [
            KPI(
                name          = kpi_names[i],
                kpi_threshold = KPI_THRESHOLDS[i],
                direction     = direction[i],
                mean          = MEAN_STD_KPIS[i][0],
                std           = MEAN_STD_KPIS[i][1],
                updatefn      = updateKPIFns[i],
            )
            for i in range(4)
        ]

        xapp_utility_fns = [compute_A1_utility, compute_A2_utility,
                            compute_A3_utility, compute_A4_utility]
        xapp_names = [f"xApp{i}" for i in range(4)]

        # xApp subscriptions match the conflict graph in Fig. 3:
        #   A1 → P1, P2
        #   A2 → P1, P3   (shares P1 with A1 → direct conflict)
        #   A3 → P4, P5
        #   A4 → P6, P7
        xapp_params = [
            (self.params[0], self.params[1]),
            (self.params[0], self.params[2]),
            (self.params[3], self.params[4]),
            (self.params[5], self.params[6]),
        ]

        self.xapps = [
            XApp(
                threshold  = KPI_THRESHOLDS[i],
                utility_fn = xapp_utility_fns[i],
                name       = xapp_names[i],
                params     = xapp_params[i],
                direction  = direction[i],
                mean_std   = MEAN_STD_KPIS[i],
            )
            for i in range(4)
        ]

        self.num_params = len(self.params)
        self.num_kpis   = len(self.kpis)
        length_state    = self.num_params + self.num_kpis

        # ── True adjacency matrix ────────────────────────────────────────────
        # Node index : 0-6 = P1-P7,  7 = K1, 8 = K2, 9 = K3, 10 = K4
        self.true_adj_matrix = np.zeros((length_state, length_state), dtype=np.float32)

        # K1 ← P1, P2
        self.true_adj_matrix[7, 0] = 1
        self.true_adj_matrix[7, 1] = 1

        # K2 ← P1, P3
        self.true_adj_matrix[8, 0] = 1
        self.true_adj_matrix[8, 2] = 1

        # K3 ← P4, P5, K1
        self.true_adj_matrix[9, 3] = 1
        self.true_adj_matrix[9, 4] = 1
        self.true_adj_matrix[9, 7] = 1

        # K4 ← P6, P7, K2
        self.true_adj_matrix[10, 5] = 1
        self.true_adj_matrix[10, 6] = 1
        self.true_adj_matrix[10, 8] = 1

        self.weights = [1] * self.num_kpis
        self.zeta    = 1e2

        self.num_bins             = num_bins
        self.action_dim           = 3
        self.min_bin_length       = int(np.min([p[1] - p[0] for p in self.paramThresholds]) // self.num_bins)
        self.bin_length_per_param = [int((p[1] - p[0]) // self.num_bins) for p in self.paramThresholds]
        self.action_space         = [self.num_params - 1, self.num_bins - 1] + self.bin_length_per_param

        self.max_steps   = max_steps
        self.cur_step    = 0
        self.prev_params = None
        self.prev_kpis   = None

        self.reset()

    def get_save_information(self):
        return {"true_graph": self.true_adj_matrix}

    def reset(self):
        self.cur_step = 0
        for p in self.params:
            low, high = p.get_threshold()
            value = np.random.uniform(low, high)
            if low == high:
                value = low
            p.set_param(value, self.params)
        self.prev_params = [p.get_param() for p in self.params]
        self.prev_kpis   = [0.0 for _ in self.kpis]
        return self._get_state()

    def reward(self, new_kpis, nw_kpis):
        return RewardFn(nw_kpis, None)

    def step(self, action: Tuple[int, int, int]):
        param_id, bin_id, index = int(action[0]), int(action[1]), int(action[2])

        low, high = self.paramThresholds[param_id]
        offset    = index

        value = low + (high - low) * (bin_id / (self.num_bins - 1)) + offset
        self.params[param_id].set_param(value, self.params)
        new_params = [p.get_param() for p in self.params]

        new_kpis = []
        for kpi in self.kpis:
            val = kpi.update_kpi(self.prev_params, self.prev_kpis)
            new_kpis.append(val)

        reward           = self.reward(self.kpis, new_kpis)
        self.prev_params = new_params
        self.prev_kpis   = new_kpis
        self.cur_step   += 1
        done = self.cur_step >= self.max_steps

        return self._get_state(), reward, done, {"success": False}

    def _get_state(self):
        state = {}
        for i, val in enumerate(self.prev_params):
            low, high = self.params[i].get_threshold()
            normalized = (val - low) / (high - low) if high != low else 1.0
            state[f"param{i}"] = np.array([normalized], dtype=np.float32)
        for kpi in self.kpis:
            val = kpi.compute_utility_value() + np.random.normal(0, 0.01)
            state[kpi.name] = np.array([val], dtype=np.float32)
        return state

    def get_state(self):
        return self._get_state()

    def observation_spec(self):
        return self._get_state()

    def observation_dims(self):
        dims = {}
        for i in range(self.num_params):
            dims[f"param{i}"] = np.array([1])
        for kpi in self.kpis:
            dims[kpi.name] = np.array([1])
        return dims

    def get_state_dim(self):
        return self.num_kpis + self.num_params

    def get_action_dim(self):
        return self.action_dim

    def action_to_param(self, action):
        param_id, bin_id, index = action
        low, high = self.paramThresholds[param_id]
        base  = low + (high - low) * (bin_id / (self.num_bins - 1))
        value = np.clip(base + index, low, high)
        return param_id, value

    def get_utility_fns(self):
        fns = []
        for i in range(len(self.xapps)):
            def make_fn(i):
                def fn(params):
                    kpis = [0.0] * self.num_kpis
                    for j, kpi in enumerate(self.kpis):
                        kpis[j] = kpi.updatefn(params, kpis)
                    return self.xapps[i].compute_utility(kpis)
                return fn
            fns.append(make_fn(i))
        return fns

    def get_thresholds_stds(self):
        return KPI_THRESHOLDS, MEAN_STD_KPIS

    def get_kpi_to_xapp_mapping(self):
        mapping = {}
        for i in range(self.num_kpis):
            mapping[self.num_params + i] = i
        return mapping