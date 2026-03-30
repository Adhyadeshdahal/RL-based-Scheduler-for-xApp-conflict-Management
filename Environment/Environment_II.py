import numpy as np
import gym
from typing import List, Callable, Tuple
from math import exp
KPI_THRESHOLDS = [55, 95, 85, 75, 80, -25]
MEAN_STD_KPIS = [(21.066, 27.599), (26.213, 34.671), 
                       (72.769, 39.040), (30.815, 40.555), 
                       (39.930, 52.155), (-17.803, 12.519)]

class XApp:
     def __init__(self,threshold,utility_fn,name,params,direction):
          self.thresholds = threshold
          self.threshold = threshold
          self.compute_utility = utility_fn
          self.name = name
          self.params = params
          self.direction = direction
          

def compute_utility_value(value,mean_std):
    mean,std = mean_std
    value = (value - mean) / std
    return value

def compute_Xapp1_utility(kpis):
    return compute_utility_value(kpis[0],MEAN_STD_KPIS[0])

def compute_Xapp2_utility(kpis):
    return compute_utility_value(kpis[1],MEAN_STD_KPIS[1])

def compute_Xapp3_utility(kpis):
    return compute_utility_value(kpis[2],MEAN_STD_KPIS[2])

def compute_Xapp4_utility(kpis):
    return compute_utility_value((kpis[3]+kpis[4])/2,MEAN_STD_KPIS[3])

def compute_Xapp5_utility(kpis):
    return compute_utility_value(kpis[5],MEAN_STD_KPIS[5])

          
def RewardFn(kpis,action):
     return sum(kpis)

     
    #  return abs(compute_Xapp1_utility()*compute_Xapp2_utility()*compute_Xapp3_utility()*compute_Xapp4_utility()*abs(compute_Xapp5_utility()))

class Param:
    def __init__(self, threshold: Tuple[float, float],id,set_param_fn):
        self.threshold = threshold
        self.id = id
        self.value = np.random.uniform(*threshold)
        self.set_param_fn = set_param_fn

    def get_param(self):
        return self.value

    def set_param(self, value,  params):
        self.value = self.set_param_fn(self.threshold,value,params)

    def get_threshold(self):
        return self.threshold
    

         

def set_param1(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param2(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param3(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param4(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param5(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param6(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param7(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value

def set_param8(threshold,value,params):
        low, high = threshold
        value = np.clip(value, low, high)

        return value


class KPI:
    def __init__(self, name: str, kpi_threshold:float,direction:int,mean:float,std:float,updatefn: Callable):
        self.name = name
        self.updatefn = updatefn
        self.value = 0.0
        self.kpi_threshold = kpi_threshold
        self.direction = direction
        self.mean = mean
        self.std = std

    def update_kpi(self, prev_params: List[float], prev_kpis: List[float]):
        self.value = self.updatefn(prev_params, prev_kpis)
        return self.value
    
    def compute_utility_value(self):
        value = (self.value - self.mean) / self.std
        return value
    
    def compute_utility_threshold(self):
        value = (self.kpi_threshold - self.mean) / self.std
        return value

    def get_kpi(self):
        return self.value

def safe_exp(x):
    return x if abs(x) > 1e-1 else 1e-1

def update_Kpi1(prev_params, prev_kpis):
    P1 = prev_params[0]
    P2 = safe_exp(prev_params[1])
    
    return 80 * exp(-(P1+50) ** 2 / (2 * (P2 ** 2)))
    # return 10 * exp(-(P1+1) ** 2 / (2 * (P2 ** 2)))



def update_Kpi2(prev_params, prev_kpis):
    P1 = prev_params[0]
    P3 = prev_params[2]
    P2 = safe_exp(prev_params[1])

    return 100*exp(-(P1 + P3) ** 2 / (2 * (P2 ** 2)))
    # return 10*exp(-(P1 + P3) ** 2 / (2 * (P2 ** 2)))



def update_Kpi3(prev_params, prev_kpis):
    P4 = safe_exp(prev_params[3])
    P1 = prev_params[0]
    return 120*exp(-(P1+45) ** 2 / (2 * (P4 ** 2)))
    # return 10*exp(-(P1+0.45) ** 2 / (2 * (P4 ** 2)))



def update_Kpi41(prev_params, prev_kpis):
    P6 = prev_params[5]
    P2 = prev_params[1]
    P5 = safe_exp(prev_params[4])
    return 120*exp(-(P6 + P2 - 30) ** 2 / (2 * (P5 ** 2)))
    # return 10*exp(-(P6 + P2-0.3) ** 2 / (2 * (P5 ** 2)))


def update_Kpi42(prev_params, prev_kpis):
    P6 = prev_params[5]
    P2 = prev_params[1]
    P5 = safe_exp(prev_params[4])
    return 150 * exp(-(P6+P2-50) ** 2 / (2 * (P5 ** 2)))
    # return 10 * exp(-(P6+P2-0.5) ** 2 / (2 * (P5 ** 2)))


def update_Kpi5(prev_params, prev_kpis):
    P7 = safe_exp(prev_params[6])
    P1 = prev_params[0]
    P8 = prev_params[7]
    return -35 * exp(-(P8 + P1 -25) ** 2 / (2 * (P7 ** 2)))
    # return -10 * exp(-(P8 + P1 - 0.25) ** 2 / (2 * (P7 ** 2)))


class ORANEnvironment2(gym.Env):

    def __init__(self, num_bins=10, max_steps=50):


        # K1         21.066     27.599      0.000     80.000  >=    55    18.7%
        # K2         26.213     34.671      0.000    100.000  >=    95     6.8%
        # K3         72.769     39.040      6.486    120.000  >=    85    49.6%
        # K41        30.815     40.555      0.000    120.000  >=    75    19.0%
        # K42        39.930     52.155      0.000    150.000  >=    80    24.6%
        # K5        -17.803     12.519    -35.000     -0.040  <=   -25    37.9%


        super().__init__()

        setParamFns = [set_param1,set_param2,set_param3,set_param4,set_param5,set_param6,set_param7,set_param8]
        self.paramThresholds = ParamThresholds = [(-100,100),(-10,50),(-20,20),(-60,60),(-20,20),(-50,150),(-60,65),(-100,150)]
        # ParamThresholds = [(0,3),(0,3),(0,3),(0,3),(0,3),(0,3),(0,3),(0,3)]

        kpi_thresholds = KPI_THRESHOLDS
        # kpi_thresholds = [5, 5, 5, 5, 5, -5]

        direction = [0, 0, 0, 0, 0, 1] # 0 means maximize, 1 means minimize
        updateKPIFns = [update_Kpi1, update_Kpi2, update_Kpi3, update_Kpi41, update_Kpi42, update_Kpi5]
        meanStdKPIs = MEAN_STD_KPIS
        xapp_utility_fns = [compute_Xapp1_utility,compute_Xapp2_utility,compute_Xapp3_utility,compute_Xapp4_utility,compute_Xapp5_utility]
        xapp_thresholds = KPI_THRESHOLDS
        xapp_names = [f"xApp{i}" for i in range(len(xapp_utility_fns))]



        # ----- Parameters -----
        self.params = [
             Param(ParamThresholds[i], i, setParamFns[i]) for i in range(8)
        ]
        
        kpi_names = ["kpi1", "kpi2", "kpi3", "kpi41", "kpi42", "kpi5"]

        # ----- KPIs -----
        #direction = 0 means maximize, direction = 1 means minimize
        self.kpis = [
            KPI(name=kpi_names[i], kpi_threshold=kpi_thresholds[i], direction=direction[i], 
                mean=meanStdKPIs[i][0], std=meanStdKPIs[i][1], updatefn=updateKPIFns[i]) 
            for i in range(6)
        ]

        xapp_params = [(self.params[0],self.params[1]),
                       (self.params[0],self.params[1],self.params[2]),
                       (self.params[0],self.params[3]),
                       (self.params[4],self.params[5],self.params[1]),
                       (self.params[0],self.params[6],self.params[7])
                       ]



        self.xapps = [
             XApp(threshold=xapp_thresholds[i],utility_fn=xapp_utility_fns[i],name=xapp_names[i],params=xapp_params[i],direction=direction[i]) for i,_ in enumerate(xapp_names)
        ]

                # ---- TRUE CAUSAL GRAPH (11 x 11) ----
        # node order:
        # 0-7  = param0..param7
        # 8    = kpi1
        # 9    = kpi2
        # 10   = kpi3
        # 11   = kpi41
        # 12   = kpi42
        # 13   = kpi5
        self.num_params = len(self.params)
        self.num_kpis = len(self.kpis)


        self.weights = [1]*self.num_kpis #need this from poilcy_params.json 
        self.zeta = 1e2 #need this from policy_params.json

        length_state = self.num_params + self.num_kpis
        self.true_adj_matrix = np.zeros((length_state, length_state), dtype=np.float32)

        # P1,P2 → K1
        self.true_adj_matrix[8, 0] = 1
        self.true_adj_matrix[8, 1] = 1

        # P1,P2,P3 → K2
        self.true_adj_matrix[9, 0] = 1
        self.true_adj_matrix[9, 1] = 1
        self.true_adj_matrix[9, 2] = 1

        # P4,P1 → K3
        self.true_adj_matrix[10, 3] = 1
        self.true_adj_matrix[10, 0] = 1

        # P2,P5,P6 → K41
        self.true_adj_matrix[11, 1] = 1
        self.true_adj_matrix[11, 4] = 1
        self.true_adj_matrix[11, 5] = 1
        
        # P2,P5,P6 → K42
        self.true_adj_matrix[12, 1] = 1
        self.true_adj_matrix[12, 4] = 1
        self.true_adj_matrix[12, 5] = 1

        #P1,P7,P8 -> K5
        self.true_adj_matrix[13, 0] = 1
        self.true_adj_matrix[13, 6] = 1
        self.true_adj_matrix[13, 7] = 1



        # ----- Discrete Action Space -----
        # action = param_index * num_bins + bin_index
        self.num_bins = num_bins
        self.action_dim = 3
        self.min_bin_length = int(np.min([param[1]-param[0] for param in self.paramThresholds]) // self.num_bins)
        self.max_bin_length = int(np.max([param[1]-param[0] for param in self.paramThresholds]) // self.num_bins)
        self.action_space = [self.num_params-1, self.num_bins-1, self.max_bin_length-1]

        self.max_steps = max_steps
        self.cur_step = 0

        self.prev_params = None
        self.prev_kpis = None

        self.reset()

    def get_save_information(self):
        return {
            "true_graph": self.true_adj_matrix
        }

    def reset(self):

        self.cur_step = 0

        for p in self.params:
            low, high = p.get_threshold()
            value = np.random.uniform(low, high)
            if low == high:
                value = low
            p.set_param(value,self.params)

        self.prev_params = [p.get_param() for p in self.params]
        self.prev_kpis = [0.0 for _ in self.kpis]

        return self._get_state()
    

    def reward(self,new_kpis, nw_kpis):
        # return sum([kpi.get_kpi() for kpi in new_kpis])
        return RewardFn(nw_kpis,None)

        satisfied_kpis = [
            kpi.value >= kpi.kpi_threshold if kpi.direction == 0 
            else kpi.value <= kpi.kpi_threshold 
            for kpi in new_kpis
        ]
        reward = sum(satisfied_kpis) ** 2

        distance = [
            (kpi.compute_utility_value() - kpi.compute_utility_threshold()) if kpi.direction == 0 
            else (kpi.compute_utility_threshold() - kpi.compute_utility_value()) 
            for kpi in new_kpis
        ]
        # distance = [
        #     (kpi.value - kpi.kpi_threshold) if kpi.direction == 0 
        #     else (kpi.kpi_threshold - kpi.value) 
        #     for kpi in new_kpis
        # ]
        distance = [max(0, d) * w for d, w in zip(distance, self.weights)]
        reward = reward - self.zeta * sum(distance)
        return reward


    def step(self, action: Tuple[int,int,int]):

        param_id, bin_id, index = action[0], action[1], action[2]
        param_0, param_1 = self.paramThresholds[param_id]
        bin_length = (param_1 - param_0) / self.num_bins  # actual bin length for THIS param
    
        # scale index from [0, num_bins-1] → [0, bin_length]
        offset = (index / (self.num_bins - 1)) * bin_length

        low, high = self.params[param_id].get_threshold()
        value = low + (high - low) * (bin_id / (self.num_bins - 1)) + offset
        # value = low + (high - low) * (bin_id / (self.num_bins - 1))

        self.params[param_id].set_param(value, self.params)
        new_params = [p.get_param() for p in self.params]

        # Update KPIs
        new_kpis = []
        for kpi in self.kpis:
            val = kpi.update_kpi(self.prev_params, self.prev_kpis)
            new_kpis.append(val)

        # reward = self.reward(self.kpis)  # uses kpi.value internally
        reward = self.reward(self.kpis,new_kpis) 
        self.prev_params = new_params
        self.prev_kpis = new_kpis
        self.cur_step += 1
        done = self.cur_step >= self.max_steps

        return self._get_state(), reward, done, {"success": False}


    def _get_state(self):
        state = {}

        for i, val in enumerate(self.prev_params):
            # state[f"param{i}"] = np.array([val], dtype=np.float32)
            low, high = self.params[i].get_threshold()
            
            normalized = (val - low) / (high - low)
            # normalized = val
            state[f"param{i}"] = np.array([normalized], dtype=np.float32)

        for kpi in self.kpis:
            val = kpi.compute_utility_value()+np.random.normal(0, 0.01) # add small noise to utility value to make it more realistic
            # val = kpi.value
            name = kpi.name
            state[name] = np.array([val], dtype=np.float32)

        return state

    def observation_spec(self):
        return self._get_state()

    def observation_dims(self):
        dims = {}
        for i in range(self.num_params):
            dims[f"param{i}"] = np.array([1])
        for kpi in self.kpis:
                name = kpi.name
                dims[name] = np.array([1])
        return dims
    
    def get_state_dim(self):
        return self.num_kpis+self.num_params
    
    def get_action_dim(self):
        return self.action_dim