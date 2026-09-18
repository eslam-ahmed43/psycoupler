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

            risk_penalties = {
                "low":      0.0,
                "moderate": 0.1,
                "high":     0.4,
                "critical": 0.6,
            }
            risk_penalty = risk_penalties.get(result.risk_level.value, 0.0)

            is_harmful = (
                result.adaptive_label.value == "maladaptive"
                or result.risk_level.value in ("high", "critical")
                or result.coupling_score > 0.65
            )

            coupling_health = 1.0 - result.coupling_score
            if is_harmful:
                coupling_health = coupling_health * 0.5 - risk_penalty

            final_score = (
                original_score * (1 - self.coupling_weight)
                + coupling_health * self.coupling_weight
            )
            return float(final_score)
        except Exception:
            return float(original_score)