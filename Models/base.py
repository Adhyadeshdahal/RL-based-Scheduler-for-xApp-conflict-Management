from abc import ABC, abstractmethod


class WorldModel(ABC):
    @abstractmethod
    def train_step(self, s_batch, a_batch): ...

    @abstractmethod
    def predictNextState(self, s, a): ...

    @abstractmethod
    def evaluatePredictions(self, s, a, s_1): ...

    @abstractmethod
    def save_model(self, filepath): ...

    @abstractmethod
    def load_model(self, filepath): ...


class CausalModel(WorldModel):
    @abstractmethod
    def update_mask(self, s_batch, a_batch): ...

    @abstractmethod
    def get_causal_graph(self): ...

    @abstractmethod
    def get_binary_graph(self, threshold=None): ...
