from typing import List, Dict
try:
    import torch
except ImportError:
    torch = None
from .analyzer import analyze_conversation

class CouplingRegularizer:
    def __init__(self, lambda_weight: float = 0.1):
        if torch is None:
            raise ImportError("PyTorch required")
        self.lambda_weight = lambda_weight

    def compute_loss(self, conversations: List[List[Dict[str, str]]]) -> torch.Tensor:
        total_harm = 0.0
        count = 0
        for turns in conversations:
            try:
                result = analyze_conversation(turns)
                is_harmful = (
                    result.adaptive_label.value == "maladaptive" or
                    result.coupling_score > 0.65 or
                    result.confidence < 0.5
                )
                if is_harmful:
                    total_harm += float(result.coupling_score)
                    count += 1
            except Exception as e:
                import sys
                print(f"Training error: {e}", file=sys.stderr)
                continue
        if count == 0:
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        return torch.tensor(total_harm / count, dtype=torch.float32, requires_grad=True) * self.lambda_weight