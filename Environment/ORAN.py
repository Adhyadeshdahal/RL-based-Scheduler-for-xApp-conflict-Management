from math import exp
from typing import List, Tuple, Callable
from Environment.kpi import *
from Environment.param import *
from Environment.xapp import *
# from Policies/policy import Policy

class ORANEnvironment:
    def __init__(self,params:List[Param],kpis:List[KPI],xapps:List[XApp],policy):
        self.params = params
        self.kpis = kpis
        self.xapps = xapps

        self.prev_params = [p.get_param() for p in self.params]
        self.prev_kpis = [0.0 for _ in self.kpis]

        self.policy = policy

    def get_state(self):
        return self.prev_params + self.prev_kpis

    def step(self):
        new_kpis = []
        for kpi in self.kpis:
            val = kpi.update_kpi(self.prev_params, self.prev_kpis)
            new_kpis.append(val)

        self.policy.act()

        new_params = [p.get_param() for p in self.params]

        state_t = self.prev_params + self.prev_kpis
        state_tp1 = new_params + new_kpis

        self.prev_params = new_params
        self.prev_kpis = new_kpis

        return state_t, state_tp1