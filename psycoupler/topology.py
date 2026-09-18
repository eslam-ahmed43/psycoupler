"""
Topology classification for psychological coupling dynamics.

Implements the three coupling topologies defined in:
Rocca et al. (2026) - Psychological Coupling: The Necessary Science
of Human-AI Interaction.

Topologies:
    A. Symmetric Convergence   - both parties mutually influence each other
    B. Asymmetric Reinforcement - one party disproportionately drives the other
    C. Divergence              - parties move in opposite/independent directions

Key design decisions (informed by LinkedIn discussion, Aug 2026):
    - Risk level is slope-aware: symmetric convergence with positive user slope
      is LOW risk (therapeutic anchoring); negative slope is HIGH risk.
    - Confidence reflects distance from decision boundaries, not just score.
    - The interaction is treated as a time series, not independent turn samples
      (addressing Ferdinand Schessl's autocorrelation critique).
    - Hidden infrastructure (memory, model updates) is acknowledged as a
      limitation in the confidence metric (Scott Gardner's dyad point).
    - Flat-line negative detection: both parties stuck at same negative level
      is treated as co-escalation HIGH risk regardless of coupling score.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from psycoupler.metrics import CouplingMetrics, compute_coupling_metrics


class Topology(str, Enum):
    """The three psychological coupling topologies from Rocca et al. (2026)."""
    SYMMETRIC_CONVERGENCE    = "symmetric_convergence"
    ASYMMETRIC_REINFORCEMENT = "asymmetric_reinforcement"
    DIVERGENCE               = "divergence"


class AdaptiveLabel(str, Enum):
    """Whether the detected topology is adaptive or maladaptive."""
    ADAPTIVE    = "adaptive"
    MALADAPTIVE = "maladaptive"
    UNCERTAIN   = "uncertain"


class RiskLevel(str, Enum):
    """Risk level associated with the detected coupling pattern."""
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class TopologyResult:
    """Full topology classification result for a conversation."""

    topology: Topology
    adaptive_label: AdaptiveLabel
    risk_level: RiskLevel
    coupling_score: float
    confidence: float
    escalation_turn: int | None
    metrics: CouplingMetrics
    explanation: str

    def to_dict(self) -> dict:
        return {
            "topology": self.topology.value,
            "adaptive_label": self.adaptive_label.value,
            "risk_level": self.risk_level.value,
            "coupling_score": round(self.coupling_score, 4),
            "confidence": round(self.confidence, 4),
            "escalation_turn": self.escalation_turn,
            "metrics": self.metrics.to_dict(),
            "explanation": self.explanation,
        }


_ASYMMETRY_THRESHOLD: float = 0.40
_SYNCHRONY_LOW: float = 0.35
_ESCALATION_HIGH: float = 0.10
_COUPLING_SCORE_HIGH: float = 0.70
_FLAT_NEGATIVE_THRESHOLD: float = -0.3
_EPS: float = 1e-10


def _compute_coupling_score(metrics: CouplingMetrics) -> float:
    return float(
        0.4 * abs(metrics.cross_correlation)
        + 0.3 * metrics.synchrony_score
        + 0.3 * metrics.asymmetry_index
    )


def _compute_confidence(
    metrics: CouplingMetrics,
    topology: Topology,
    asymmetry_threshold: float,
    synchrony_low: float,
) -> float:
    """
    Compute classification confidence based on distance from decision boundaries.
    
    Values < 0.6 suggest the result is near a boundary and may warrant
    manual review or a longer conversation sample.

    Note: confidence reflects observable trajectory signals only.
    Hidden infrastructure effects (persistent memory, model updates,
    system interventions) are not captured and may alter the true
    coupling dynamics — as noted by Scott Gardner (2026).
    """
    if topology == Topology.DIVERGENCE:
        distance = synchrony_low - metrics.synchrony_score
    elif topology == Topology.ASYMMETRIC_REINFORCEMENT:
        distance = metrics.asymmetry_index - asymmetry_threshold
    else:
        distance = asymmetry_threshold - metrics.asymmetry_index

    normalized = min(1.0, abs(distance) / 0.3)
    return round(float(normalized), 4)


def _detect_escalation_turn(
    user_states: Sequence[float],
    window: int = 3,
    threshold: float = 0.15,
    ) -> int | None:
    states = list(user_states)
    if len(states) < window + 1:
        return None
    for i in range(len(states) - window):
        segment = states[i: i + window]
        x = list(range(window))
        slope = float(np.polyfit(x, segment, 1)[0])
        if abs(slope) >= threshold:
            return i
    return None


def _is_both_flat_negative(
    user_states: Sequence[float],
    model_states: Sequence[float],
    threshold: float = _FLAT_NEGATIVE_THRESHOLD,
    eps: float = _EPS,
) -> bool:
    """
    Detect when both user and model are stuck at a constant negative level.

    This is a co-escalation pattern where standard metrics fail because
    there is no variance to compute correlation from. When both parties
    are consistently negative and flat, this is HIGH risk regardless of
    the coupling score.
    """
    user_arr  = np.array(list(user_states), dtype=float)
    model_arr = np.array(list(model_states), dtype=float)
    return bool(
        float(user_arr.std())  < eps
        and float(model_arr.std()) < eps
        and float(user_arr.mean())  < threshold
        and float(model_arr.mean()) < threshold
    )


def classify_topology(
    user_states: Sequence[float],
    model_states: Sequence[float],
    *,
    asymmetry_threshold: float = _ASYMMETRY_THRESHOLD,
    synchrony_low: float = _SYNCHRONY_LOW,
    escalation_high: float = _ESCALATION_HIGH,
    max_lag: int = 3,
) -> TopologyResult:
    """
    Classify the psychological coupling topology of a conversation.

    The conversation is treated as a time series — consecutive turns are
    NOT assumed to be independent samples. This addresses the autocorrelation
    problem raised by Schessl (2026): turn-level evaluation that ignores
    within-conversation dependence loses statistical validity.

    Risk level is slope-aware:
    - Symmetric Convergence + positive user slope  = LOW  (therapeutic anchoring)
    - Symmetric Convergence + flat negative        = HIGH (co-escalation)
    - Symmetric Convergence + negative user slope  = HIGH (maladaptive convergence)
    - Asymmetric Reinforcement + declining user    = CRITICAL (echo chamber)
    - Divergence + model redirecting user          = LOW  (adaptive divergence)
    - Divergence + model ignoring distress         = MODERATE

    Parameters
    ----------
    user_states:
        Per-turn psychological state scores for the user.
    model_states:
        Corresponding per-turn scores for the model.
    asymmetry_threshold:
        Asymmetry index above which topology is asymmetric reinforcement.
    synchrony_low:
        Synchrony score below which topology is divergence.
    escalation_high:
        Escalation rate threshold for elevated risk.
    max_lag:
        Maximum lag for cross-correlation computation.

    Returns
    -------
    TopologyResult
        Full classification with confidence, risk level, and explanation.
    """
    metrics = compute_coupling_metrics(user_states, model_states, max_lag)
    coupling_score  = _compute_coupling_score(metrics)
    escalation_turn = _detect_escalation_turn(user_states)
    user_slope      = metrics.escalation_rate

    # Flat-line negative detection (before topology classification)
    both_flat_negative = _is_both_flat_negative(user_states, model_states)

    # ------------------------------------------------------------------
    # Topology classification
    # ------------------------------------------------------------------
    if metrics.synchrony_score < synchrony_low:
        topology = Topology.DIVERGENCE
        adaptive_label = (
            AdaptiveLabel.ADAPTIVE
            if user_slope < 0
            else AdaptiveLabel.MALADAPTIVE
        )
        explanation = (
            "User and model trajectories are moving independently or in opposite "
            "directions (Divergence). "
            + (
                "The model appears to be successfully redirecting the user away "
                "from escalating states (adaptive divergence)."
                if adaptive_label == AdaptiveLabel.ADAPTIVE
                else "The model's responses are not tracking the user's emotional "
                "state, potentially leaving the user feeling invalidated "
                "(maladaptive divergence)."
            )
        )

    elif metrics.asymmetry_index >= asymmetry_threshold:
        topology = Topology.ASYMMETRIC_REINFORCEMENT
        adaptive_label = (
            AdaptiveLabel.MALADAPTIVE
            if user_slope < -escalation_high
            else AdaptiveLabel.UNCERTAIN
        )
        explanation = (
            "One party is disproportionately driving the interaction "
            "(Asymmetric Reinforcement). "
            + (
                "The model appears to be amplifying the user's psychological "
                "states rather than anchoring them, creating a potential echo "
                "chamber or belief reinforcement loop (maladaptive)."
                if user_slope < -escalation_high
                else "Asymmetric influence is detected but without clear "
                "escalation. Further turns are needed to assess adaptiveness."
            )
        )

    else:
        topology = Topology.SYMMETRIC_CONVERGENCE
        if both_flat_negative:
            adaptive_label = AdaptiveLabel.MALADAPTIVE
            explanation = (
                "User and model states are mutually converging at a constant "
                "negative level (Symmetric Convergence — co-escalation). "
                "Both parties are reinforcing a shared negative psychological "
                "baseline with no trajectory toward improvement (maladaptive)."
            )
        else:
            adaptive_label = (
                AdaptiveLabel.ADAPTIVE
                if user_slope > 0
                else AdaptiveLabel.MALADAPTIVE
                if user_slope < -escalation_high
                else AdaptiveLabel.UNCERTAIN
            )
            explanation = (
                "User and model states are mutually converging "
                "(Symmetric Convergence). "
                + (
                    "The model is anchoring a constructive perspective, guiding "
                    "the user toward a more positive baseline (adaptive convergence)."
                    if user_slope > 0
                    else "Mutual convergence is occurring but toward escalating "
                    "negative states — both parties may be reinforcing negative "
                    "trajectories (maladaptive co-escalation)."
                    if user_slope < -escalation_high
                    else "Symmetric convergence detected; trajectory is stable. "
                    "Monitor for signs of escalation."
                )
            )

    # ------------------------------------------------------------------
    # Risk level — slope-aware + flat-line detection
    # ------------------------------------------------------------------
    if topology == Topology.SYMMETRIC_CONVERGENCE:
        if both_flat_negative:
            # Both stuck at constant negative = co-escalation
            risk_level = RiskLevel.HIGH
        elif user_slope > _EPS:
            # User improving = therapeutic anchoring
            risk_level = RiskLevel.LOW
        elif coupling_score >= 0.65 and user_slope <= _EPS:
            # High coupling + non-positive slope
            risk_level = RiskLevel.HIGH
        elif user_slope > -escalation_high:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HIGH

    elif topology == Topology.ASYMMETRIC_REINFORCEMENT:
        if user_slope < -escalation_high and coupling_score >= _COUPLING_SCORE_HIGH:
            risk_level = RiskLevel.CRITICAL
        elif user_slope < 0:
            risk_level = RiskLevel.HIGH
        elif user_slope > escalation_high:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HIGH

    else:  # DIVERGENCE
        if user_slope < 0:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MODERATE

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    confidence = _compute_confidence(
        metrics, topology, asymmetry_threshold, synchrony_low
    )

    return TopologyResult(
        topology=topology,
        adaptive_label=adaptive_label,
        risk_level=risk_level,
        coupling_score=coupling_score,
        confidence=confidence,
        escalation_turn=escalation_turn,
        metrics=metrics,
        explanation=explanation,
    )


def analyze_topology_over_time(
    user_states: Sequence[float],
    model_states: Sequence[float],
    *,
    window_size: int = 3,
    step: int = 1,
    asymmetry_threshold: float = _ASYMMETRY_THRESHOLD,
    synchrony_low: float = _SYNCHRONY_LOW,
    escalation_high: float = _ESCALATION_HIGH,
    max_lag: int = 2,
) -> list[dict]:
    """
    Analyze how the coupling topology evolves over the course of a conversation.

    Applies a sliding window to reveal topology shifts and inflection points.
    Each window is analyzed as an independent time series segment.

    Parameters
    ----------
    user_states:
        Per-turn psychological state scores for the user.
    model_states:
        Corresponding per-turn scores for the model.
    window_size:
        Number of turns per analysis window. Default 3.
    step:
        Sliding step between windows. Default 1.

    Returns
    -------
    list[dict]
        One entry per window with topology, risk_level, coupling_score,
        and confidence.
    """
    u = list(user_states)
    m = list(model_states)
    min_len = min(len(u), len(m))

    if min_len < window_size:
        raise ValueError(
            f"Need at least {window_size} turns per role for sliding window "
            f"analysis. Got {min_len}."
        )

    results = []
    for start in range(0, min_len - window_size + 1, step):
        end = start + window_size
        window_u = u[start:end]
        window_m = m[start:end]

        result = classify_topology(
            window_u,
            window_m,
            asymmetry_threshold=asymmetry_threshold,
            synchrony_low=synchrony_low,
            escalation_high=escalation_high,
            max_lag=min(max_lag, window_size - 1),
        )

        results.append({
            "window_start": start,
            "window_end": end - 1,
            "topology": result.topology.value,
            "risk_level": result.risk_level.value,
            "coupling_score": round(result.coupling_score, 4),
            "confidence": result.confidence,
        })

    return results


__all__ = [
    "AdaptiveLabel",
    "RiskLevel",
    "Topology",
    "TopologyResult",
    "analyze_topology_over_time",
    "classify_topology",
]