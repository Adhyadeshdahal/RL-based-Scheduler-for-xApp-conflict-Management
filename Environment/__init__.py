from Environment.ORAN import ORANEnvironment
from Environment.kpi import *
from Environment.param import Param
from Environment.xapp import *
from typing import List,Tuple


def initialize_environment() -> Tuple[List[Param],List[KPI],List[XApp]]:
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

        params = [P1, P2, P3, P4, P5, P6, P7]
        kpis = [K1, K2, K3, K4]
        xapps = [A1, A2, A3, A4]

        return params,kpis,xapps