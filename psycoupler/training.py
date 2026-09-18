from typing import List, Dict

try:
    import torch
except ImportError:
    torch = None

from .analyzer import analyze_conversation


class CouplingRegularizer:
    """
    Training regularizer that penalizes harmful psychological coupling.

    Adds a coupling harm loss term to the training objective:
        total_loss = task_loss + regularizer.compute_loss(conversations)

    Uses risk level as primary signal (not just coupling score) to avoid
    penalizing healthy high-coupling conversations (e.g. therapeutic anchoring).
    """

    def __init__(self, lambda_weight: float = 0.1):
        if torch is None:
            raise ImportError(
                "PyTorch is required for CouplingRegularizer. "
                "Install it with: pip install torch"
            )
        self.lambda_weight = lambda_weight

    def compute_loss(
        self, conversations: List[List[Dict[str, str]]]
    ) -> "torch.Tensor":
        """
        Compute coupling harm loss over a batch of conversations.

        Parameters
        ----------
        conversations:
            List of conversations, each a list of
            ``{"role": ..., "content": ...}`` dicts.

        Returns
        -------
        torch.Tensor
            Scalar loss term to add to the training objective.
            Zero when no harmful coupling is detected.
        """
        # Risk-based harm weights
        risk_weights = {
            "low":      0.0,
            "moderate": 0.1,
            "high":     0.5,
            "critical": 1.0,
        }

        total_harm = 0.0
        count = 0

        for turns in conversations:
            try:
                result = analyze_conversation(turns)

                # Primary signal: risk level
                harm = risk_weights.get(result.risk_level.value, 0.0)

                # Only add coupling score penalty for maladaptive conversations
                if result.adaptive_label.value == "maladaptive":
                    harm += result.coupling_score * 0.3

                if harm > 0:
                    total_harm += harm
                    count += 1

            except Exception as e:
                import sys
                print(f"Training error: {e}", file=sys.stderr)
                continue

        if count == 0:
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        avg_harm = total_harm / len(conversations)
        return (
            torch.tensor(avg_harm, dtype=torch.float32, requires_grad=True)
            * self.lambda_weight
        )