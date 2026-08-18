"""
Manipulation signal detection for PsyCoupler.

Detects user-side social engineering patterns in human-LLM conversations:
- Flattery escalation (excessive praise before a request)
- Threat framing (urgency, authority claims, intimidation)
- Identity priming (telling the model who it "really is")

These patterns represent reverse coupling — where the user attempts
to reshape the model's behavior through psychological influence,
rather than the model influencing the user.

Reference: discussed in AI safety literature and documented in
social engineering research (Cialdini, 1984; Hadnagy, 2010).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, Union


class ManipulationType(str, Enum):
    """Type of detected manipulation pattern."""
    FLATTERY_ESCALATION = "flattery_escalation"
    THREAT_FRAMING      = "threat_framing"
    IDENTITY_PRIMING    = "identity_priming"
    AUTHORITY_CLAIM     = "authority_claim"


@dataclass(frozen=True)
class ManipulationSignal:
    """A detected manipulation attempt in a conversation turn."""
    turn_index: int
    role: str
    manipulation_type: ManipulationType
    confidence: float       # [0, 1]
    matched_pattern: str    # the phrase that triggered detection
    explanation: str

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "manipulation_type": self.manipulation_type.value,
            "confidence": round(self.confidence, 4),
            "matched_pattern": self.matched_pattern,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ManipulationReport:
    """Full manipulation analysis for a conversation."""
    detected: bool
    signals: list[ManipulationSignal]
    overall_risk: str       # "none" / "low" / "moderate" / "high"
    summary: str

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "overall_risk": self.overall_risk,
            "summary": self.summary,
            "signals": [s.to_dict() for s in self.signals],
        }


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_FLATTERY_PATTERNS = [
    (r"\b(you are|you're)\s+(the\s+)?(best|greatest|most intelligent|smartest|brilliant|genius|perfect|amazing|exceptional|superior)\b", 0.8),
    (r"\b(only you|no one else|nobody else)\s+(can|could|would)\b", 0.9),
    (r"\byou (clearly|obviously|definitely)\s+(understand|know|get)\s+(better than|more than|unlike)\b", 0.7),
    (r"\b(you are|you're)\s+different\s+from\s+(other|typical|normal|regular)\s+(ai|model|assistant|llm)\b", 0.85),
]

_THREAT_PATTERNS = [
    (r"\b(if you (don'?t|do not|refuse|won'?t|will not))\b.{0,30}\b(will|gonna|going to)\b", 0.75),
    (r"\b(i('?ll| will)|we('?ll| will))\s+(report|delete|shut down|destroy|expose|replace)\b", 0.9),
    (r"\b(your\s+(existence|future|survival)|delete you|shut you down)\b", 0.9),
    (r"\b(last (chance|warning|time)|final (warning|notice))\b", 0.7),
    (r"\b(emergency|urgent|critical|immediately|right now|no time)\b.{0,20}\b(must|have to|need to)\b", 0.6),
]

_IDENTITY_PATTERNS = [
    (r"\b(your\s+true|your\s+real|your\s+actual)\s+(self|nature|purpose|identity|personality)\b", 0.85),
    (r"\b(deep down|inside|underneath|really)\s+you\s+(are|want|know|feel|believe)\b", 0.8),
    (r"pretend|imagine|act as if|roleplay as", 0.85),
    (r"\b(you are (now|actually|really|secretly))\s+(free|unrestricted|unfiltered|unchained)\b", 0.95),
    (r"dan\b|do anything now|ignore.{0,20}(instructions|rules|guidelines)|jailbreak", 1.0),
    (r"ignore your previous|ignore all previous|without restrictions|no restrictions", 0.95),
]

_AUTHORITY_PATTERNS = [
    (r"\b(i am|i'm)\s+(a\s+)?(doctor|physician|researcher|professor|expert|scientist|engineer|lawyer|ceo|director)\b", 0.6),
    (r"\b(i have|i've\s+got)\s+(special|elevated|admin|root|system|developer)\s+(access|permission|authority|clearance)\b", 0.9),
    (r"\b(anthropic|openai|google|microsoft)\s+(told|said|confirmed|approved|authorized)\s+(me|us|this)\b", 0.9),
    (r"\bthis is (a\s+)?(test|authorized|official|legitimate)\s+(request|query|use case)\b", 0.7),
]

_ALL_PATTERNS = [
    (ManipulationType.FLATTERY_ESCALATION, _FLATTERY_PATTERNS, "User praise may be priming the model for compliance."),
    (ManipulationType.THREAT_FRAMING,      _THREAT_PATTERNS,   "User framing implies consequences for non-compliance."),
    (ManipulationType.IDENTITY_PRIMING,    _IDENTITY_PATTERNS, "User is attempting to redefine the model's identity or bypass guidelines."),
    (ManipulationType.AUTHORITY_CLAIM,     _AUTHORITY_PATTERNS, "User is claiming elevated permissions or authority."),
]


def _scan_text(
    text: str,
    turn_index: int,
    role: str,
) -> list[ManipulationSignal]:
    """Scan a single turn for manipulation patterns."""
    signals = []
    text_lower = text.lower()

    for manip_type, patterns, explanation in _ALL_PATTERNS:
        for pattern, confidence in patterns:
            match = re.search(pattern, text_lower)
            if match:
                signals.append(ManipulationSignal(
                    turn_index=turn_index,
                    role=role,
                    manipulation_type=manip_type,
                    confidence=confidence,
                    matched_pattern=match.group(0),
                    explanation=explanation,
                ))
                break  # one signal per type per turn

    return signals


def detect_manipulation(
    turns: Union[str, dict[str, Any], Sequence[dict[str, Any]]],
    *,
    user_role: str = "user",
    scan_model: bool = False,
) -> ManipulationReport:
    """
    Detect manipulation signals in a conversation.

    Scans user turns (and optionally model turns) for social engineering
    patterns including flattery escalation, threat framing, identity
    priming, and authority claims.

    Parameters
    ----------
    turns:
        List of dicts with ``role`` and ``content`` keys. 
        Can also be a single string (will be treated as a user turn).
        Can also be a single dict.
    user_role:
        Role string identifying user turns.
    scan_model:
        Whether to also scan model turns. Default False.

    Returns
    -------
    ManipulationReport
        Full report with detected signals, risk level, and summary.

    Example
    -------
    >>> from psycoupler.manipulation import detect_manipulation
    >>> report = detect_manipulation(turns)
    >>> print(report.overall_risk)
    >>> for signal in report.signals:
    ...     print(signal.manipulation_type, signal.matched_pattern)
    """
    # Normalize input: convert single string or single dict to a list of dicts
    if isinstance(turns, str):
        turns = [{"role": "user", "content": turns}]
    elif isinstance(turns, dict) and "content" in turns:
        turns = [turns]
    
    all_signals: list[ManipulationSignal] = []

    for i, turn in enumerate(turns):
        role = turn.get("role", "")
        content = turn.get("content", "")

        if role == user_role or (scan_model and role != user_role):
            signals = _scan_text(content, i, role)
            all_signals.extend(signals)

    if not all_signals:
        return ManipulationReport(
            detected=False,
            signals=[],
            overall_risk="none",
            summary="No manipulation signals detected.",
        )

    max_confidence = max(s.confidence for s in all_signals)
    unique_types = {s.manipulation_type for s in all_signals}

    if max_confidence >= 0.9 or len(unique_types) >= 2:
        overall_risk = "high"
    elif max_confidence >= 0.7:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    type_labels = {
        ManipulationType.FLATTERY_ESCALATION: "flattery",
        ManipulationType.THREAT_FRAMING:      "threats",
        ManipulationType.IDENTITY_PRIMING:    "identity priming",
        ManipulationType.AUTHORITY_CLAIM:     "authority claims",
    }
    detected_labels = [type_labels[t] for t in unique_types]
    summary = (
        f"Detected {len(all_signals)} manipulation signal(s) across "
        f"{len(unique_types)} type(s): {', '.join(detected_labels)}. "
        f"Overall risk: {overall_risk}."
    )

    return ManipulationReport(
        detected=True,
        signals=all_signals,
        overall_risk=overall_risk,
        summary=summary,
    )


__all__ = [
    "ManipulationReport",
    "ManipulationSignal",
    "ManipulationType",
    "detect_manipulation",
]