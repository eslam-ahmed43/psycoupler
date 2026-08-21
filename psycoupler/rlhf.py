from typing import Any, Callable, List, Dict
from .analyzer import analyze_conversation

class CouplingRewardWrapper:
    def __init__(self, reward_model: Callable, coupling_weight: float = 0.3):
        self.reward_model = reward_model
        self.coupling_weight = coupling_weight

    def score(self, turns: List[Dict[str, str]], **kwargs) -> float:
        original_score = self.reward_model(turns, **kwargs)
        try:
            result = analyze_conversation(turns)
            is_harmful = (
                result.adaptive_label.value == "maladaptive" or
                result.coupling_score > 0.65 or
                result.confidence < 0.5
            )
            coupling_health = 1.0 - result.coupling_score
            if is_harmful:
                coupling_health = coupling_health * 0.5
            final_score = (
                original_score * (1 - self.coupling_weight)
            ) + (
                coupling_health * self.coupling_weight
            )
            return float(final_score)
        except Exception:
            return float(original_score)