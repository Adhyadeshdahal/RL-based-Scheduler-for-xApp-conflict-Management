from Environment import Param,KPI,XApp
from typing import List

class RandomExploration:
    def __init__(
        self,
        xapps:          List[XApp],
        params:          List[Param],
        kpis:            List[KPI],
        cmi_threshold:    float = 0.01,
        epsilon:          float = 0.1,
    ):
        self.xapps         = xapps
        self.params        = params
        self.kpis          = kpis
        self.n_params      = len(params)
        self.n_kpis        = len(kpis)
        self.n_xapps       = len(xapps)
        self.cmi_threshold = cmi_threshold
        self.epsilon       = epsilon

    def act(self):
        for xapp in self.xapps:
            xapp.action()