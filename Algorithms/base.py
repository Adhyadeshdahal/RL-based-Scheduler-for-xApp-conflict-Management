from abc import ABC, abstractmethod


class Planner(ABC):
    name: str

    @abstractmethod
    def act(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ) -> list[int]: ...
