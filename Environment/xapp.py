from typing import List,Callable
from Environment.param import Param
from Environment.kpi import  KPI
import random

class XApp:
    def __init__(self, param: List[Param], kpi: List[KPI], action_fn: Callable):
        self.params_controlled = param
        self.kpis_monitored = kpi
        self.action_fn = action_fn
    
    def action(self):
        return self.action_fn(self)

#selects and sets the new value
def action_fn(self):
    for param in self.params_controlled:
        low, high = param.get_threshold()
        new_value = random.uniform(low, high)
        param.set_param(new_value)