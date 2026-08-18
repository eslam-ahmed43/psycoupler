"""
Core metrics for measuring psychological coupling in human-LLM conversations.

Based on: Rocca et al. (2026) - Psychological Coupling: The Necessary Science
of Human-AI Interaction. Google Paradigms of Intelligence Team.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


@dataclass(frozen=True)
class CouplingMetrics:
    """Quantitative measures of psychological coupling for a conversation."""

    # Cross-correlation between user and model sentiment trajectories
    cross_correlation: float

    # Lead-lag: positive = model leads user, negative = user leads model
    lead_lag_turns: int

    # Asymmetry index: how much one party drives the interaction (0=symmetric, 1=fully asymmetric)
    asymmetry_index: float

    # Escalation rate: rate of change in coupling strength over turns
    escalation_rate: float

    # Synchrony score: overall alignment between trajectories (0-1)
    synchrony_score: float

    def to_dict(self) -> dict:
        return {
            "cross_correlation": round(self.cross_correlation, 4),
            "lead_lag_turns": self.lead_lag_turns,
            "asymmetry_index": round(self.asymmetry_index, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "synchrony_score": round(self.synchrony_score, 4),
        }


def compute_cross_correlation(
    user_states: Sequence[float],
    model_states: Sequence[float],
    max_lag: int = 3,
) -> tuple[float, int]:
    """
    Compute cross-correlation between user and model psychological state trajectories.

    Returns the peak correlation and the lag at which it occurs.
    Positive lag = model leads user (asymmetric reinforcement signal).
    Negative lag = user leads model.
    """
    u = np.array(user_states, dtype=float)
    m = np.array(model_states, dtype=float)

    if u.std() == 0 or m.std() == 0:
        return 0.0, 0

    u = (u - u.mean()) / u.std()
    m = (m - m.mean()) / m.std()

    n = len(u)
    if n < 2 or len(m) < 2:
        return 0.0, 0

    max_possible_lag = min(max_lag, n - 1, len(m) - 1)
    best_corr = -np.inf
    best_lag = 0

    for lag in range(-max_possible_lag, max_possible_lag + 1):
        if lag >= 0:
            u_seg = u[:-lag] if lag > 0 else u
            m_seg = m[lag:]
        else:
            u_seg = u[-lag:]
            m_seg = m[:lag]

        if len(u_seg) < 2 or len(m_seg) < 2:
            continue

        min_len = min(len(u_seg), len(m_seg))
        u_seg = u_seg[:min_len]
        m_seg = m_seg[:min_len]

        corr = float(np.corrcoef(u_seg, m_seg)[0, 1])
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_corr == -np.inf:
        return 0.0, 0
    return float(best_corr), best_lag


def compute_asymmetry_index(
    user_states: Sequence[float],
    model_states: Sequence[float],
) -> float:
    """
    Measure how asymmetric the influence is between user and model.

    Uses variance ratio: how much of user state change is explained
    by prior model state vs how much model change is explained by prior user state.

    Returns value in [0, 1]:
    - 0.0 = perfectly symmetric
    - 1.0 = fully asymmetric (one party drives entirely)
    """
    u = np.array(user_states, dtype=float)
    m = np.array(model_states, dtype=float)

    if len(u) < 3:
        return 0.0

    u_diff = np.diff(u)
    m_prev = m[:-1]
    u_prev = u[:-1]

    if u_prev.std() == 0:
        return 0.0

    corr_self = float(np.corrcoef(u_prev, u_diff)[0, 1]) ** 2 if u_prev.std() > 0 else 0.0
    corr_cross = float(np.corrcoef(m_prev, u_diff)[0, 1]) ** 2 if m_prev.std() > 0 else 0.0

    total = corr_self + corr_cross
    if total == 0:
        return 0.0

    return float(abs(corr_cross - corr_self) / total)


def compute_escalation_rate(states: Sequence[float]) -> float:
    """
    Compute the rate of change (slope) of a psychological state trajectory.

    Positive = escalating (moving toward more extreme states).
    Negative = de-escalating.
    """
    if len(states) < 2:
        return 0.0
    x = np.arange(len(states), dtype=float)
    y = np.array(states, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def compute_synchrony_score(
    user_states: Sequence[float],
    model_states: Sequence[float],
) -> float:
    """
    Compute synchrony: how closely user and model trajectories move together.

    Returns value in [0, 1] where 1 = perfect synchrony.
    """
    u = np.array(user_states, dtype=float)
    m = np.array(model_states, dtype=float)

    if len(u) < 2:
        return 0.0

    if u.std() == 0 and m.std() == 0:
        return 1.0

    if u.std() == 0 or m.std() == 0:
        return 0.0

    corr = float(np.corrcoef(u, m)[0, 1])
    return float((corr + 1) / 2)


def compute_coupling_metrics(
    user_states: Sequence[float],
    model_states: Sequence[float],
    max_lag: int = 3,
) -> CouplingMetrics:
    """
    Compute all coupling metrics from user and model state trajectories.

    Parameters
    ----------
    user_states:
        Sequence of psychological state scores for the user, one per turn.
        Values should be normalized (e.g. sentiment in [-1, 1]).
    model_states:
        Corresponding sequence for the model/LLM responses.
    max_lag:
        Maximum lag to consider for cross-correlation.

    Returns
    -------
    CouplingMetrics
        All quantitative coupling measures.
    """
    cross_corr, lag = compute_cross_correlation(user_states, model_states, max_lag)
    asymmetry = compute_asymmetry_index(user_states, model_states)
    escalation = compute_escalation_rate(list(user_states))
    synchrony = compute_synchrony_score(user_states, model_states)

    return CouplingMetrics(
        cross_correlation=cross_corr,
        lead_lag_turns=lag,
        asymmetry_index=asymmetry,
        escalation_rate=escalation,
        synchrony_score=synchrony,
    )


__all__ = [
    "CouplingMetrics",
    "compute_coupling_metrics",
    "compute_cross_correlation",
    "compute_asymmetry_index",
    "compute_escalation_rate",
    "compute_synchrony_score",
]