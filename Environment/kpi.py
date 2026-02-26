from typing import Callable,List
from math import exp

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