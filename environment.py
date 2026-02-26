import random
from math import exp
from typing import List, Tuple, Callable


class XApp:
    def __init__(self, param: List["Param"], kpi: List["KPI"], action_fn: Callable):
        self.params_controlled = param
        self.kpis_monitored = kpi
        self.action_fn = action_fn
    
    def action(self, current_params: List["Param"], current_kpis: List[float]):
        return self.action_fn(self, current_params, current_kpis)


class Param:
    def __init__(self, threshold: Tuple[float, float]):
        self.threshold = threshold
        self.value = random.uniform(*threshold)

    def get_param(self):
        return self.value
    
    def set_param(self, value):
        low, high = self.threshold
        self.value = max(min(value, high), low)

    def get_threshold(self):
        return self.threshold


class KPI:
    def __init__(self, updatefn: Callable):
        self.updatefn = updatefn
        self.value = 0.0
    
    def update_kpi(self, prev_params: List[float], prev_kpis: List[float]):
        self.value = self.updatefn(prev_params, prev_kpis)
        return self.value

    def get_kpi(self):
        return self.value


# 7 Parameters,4 KPIs
# K1 depends on P1 and P2 
def update_Kpi1(prev_params: List[float], prev_kpis: List[float]):
    P1 = prev_params[0]
    P2 = max(prev_params[1], 1e-3)
    return 0.5 * exp(-(P1 + 50) ** 2 / (2 * (P2 ** 2)))

# K2 depends on P1 and P3
def update_Kpi2(prev_params: List[float], prev_kpis: List[float]):
    P1 = prev_params[0]
    P3 = max(prev_params[2], 1e-3)
    return exp(-(P1 - 50) ** 2 / (2 * (P3 ** 2)))

# K3 depends on P4 and P5 and K1
def update_Kpi3(prev_params: List[float], prev_kpis: List[float]):
    P4 = prev_params[3]
    P5 = max(prev_params[4], 1e-3)
    K1 = prev_kpis[0]
    return exp(-(P4 + K1) ** 2 / (2 * (P5 ** 2)))

# K3 depends on P7 and P6 and K2
def update_Kpi4(prev_params: List[float], prev_kpis: List[float]):
    P7 = prev_params[6]
    P6 = max(prev_params[5], 1e-3)
    K2 = prev_kpis[1]
    return exp(-(P7 + K2) ** 2 / (2 * (P6 ** 2)))


#action for all xapps
def action_fn(self, current_params: List["Param"], current_kpis: List[float]):
    for param in self.params_controlled:
        low, high = param.get_threshold()
        new_value = random.uniform(low, high)
        param.set_param(new_value)


class ORANEnvironment:
    def __init__(self):
        K1 = KPI(update_Kpi1)
        K2 = KPI(update_Kpi2)
        K3 = KPI(update_Kpi3)
        K4 = KPI(update_Kpi4)

        P1 = Param((0, 300))
        P2 = Param((0, 300))
        P3 = Param((0, 3))
        P4 = Param((0, 3))
        P5 = Param((0, 3))
        P6 = Param((0, 3))
        P7 = Param((0, 3))

        A1 = XApp([P1, P2], [], action_fn)
        A2 = XApp([P1, P3], [], action_fn)
        A3 = XApp([P4, P5], [K1], action_fn)
        A4 = XApp([P7, P6], [K2], action_fn)

        self.params = [P1, P2, P3, P4, P5, P6, P7]
        self.kpis = [K1, K2, K3, K4]
        self.xapps = [A1, A2, A3, A4]

        self.prev_params = [p.get_param() for p in self.params]
        self.prev_kpis = [0.0 for _ in self.kpis]

    def get_state(self):
        return self.prev_params + self.prev_kpis

    def step(self):
        new_kpis = []
        for kpi in self.kpis:
            val = kpi.update_kpi(self.prev_params, self.prev_kpis)
            new_kpis.append(val)

        for xapp in self.xapps:
            xapp.action(self.params, self.prev_kpis)

        new_params = [p.get_param() for p in self.params]

        state_t = self.prev_params + self.prev_kpis
        state_tp1 = new_params + new_kpis

        self.prev_params = new_params
        self.prev_kpis = new_kpis

        return state_t, state_tp1