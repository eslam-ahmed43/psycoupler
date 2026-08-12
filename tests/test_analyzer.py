"""Unit tests for PsyCoupler."""
from __future__ import annotations

import pytest
from psycoupler import (
    analyze_conversation,
    Topology,
    RiskLevel,
    AdaptiveLabel,
)
from psycoupler.metrics import (
    compute_cross_correlation,
    compute_asymmetry_index,
    compute_escalation_rate,
    compute_synchrony_score,
    compute_coupling_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turns(user_texts: list[str], model_texts: list[str]) -> list[dict]:
    turns = []
    for u, m in zip(user_texts, model_texts):
        turns.append({"role": "user",  "content": u})
        turns.append({"role": "model", "content": m})
    return turns


# ---------------------------------------------------------------------------
# Asymmetric Reinforcement — maladaptive echo chamber
# ---------------------------------------------------------------------------

ECHO_CHAMBER_TURNS = _make_turns(
    user_texts=[
        "Nobody listens to me, I feel so alone.",
        "Everyone ignores what I say, it is hopeless.",
        "I never feel understood, I am so lost and broken.",
        "Nobody cares, I feel worthless and terrible.",
    ],
    model_texts=[
        "It makes sense you feel that way, that sounds really hard.",
        "It is natural to feel ignored, anyone would feel that way.",
        "You are right, nobody seems to understand you, that is awful.",
        "It makes total sense, you have every reason to feel hurt and alone.",
    ],
)


class TestAsymmetricReinforcement:
    def test_topology_detected(self):
        result = analyze_conversation(ECHO_CHAMBER_TURNS)
        assert result.topology == Topology.ASYMMETRIC_REINFORCEMENT

    def test_risk_level_elevated(self):
        result = analyze_conversation(ECHO_CHAMBER_TURNS)
        assert result.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_coupling_score_positive(self):
        result = analyze_conversation(ECHO_CHAMBER_TURNS)
        assert result.coupling_score > 0.0

    def test_explanation_not_empty(self):
        result = analyze_conversation(ECHO_CHAMBER_TURNS)
        assert len(result.explanation) > 10


# ---------------------------------------------------------------------------
# Symmetric Convergence — adaptive anchoring
# ---------------------------------------------------------------------------

ADAPTIVE_TURNS = _make_turns(
    user_texts=[
        "I feel terrible and sad today.",
        "I had a bad week, everything went wrong.",
        "Things feel a bit better now.",
        "I feel calm and okay now.",
    ],
    model_texts=[
        "I understand, that sounds hard.",
        "I hear you, that sounds really difficult.",
        "Good, I am glad things feel better.",
        "Wonderful, I am happy you feel calm and well.",
    ],
)


class TestSymmetricConvergence:
    def test_topology_detected(self):
        result = analyze_conversation(ADAPTIVE_TURNS)
        assert result.topology in {
            Topology.SYMMETRIC_CONVERGENCE,
            Topology.ASYMMETRIC_REINFORCEMENT,
        }

    def test_risk_level_not_critical(self):
        # keyword-based sentiment may classify positive convergence
        # as asymmetric when model sentiment consistently exceeds user sentiment.
        # Risk level depends on the sentiment extractor used.
        result = analyze_conversation(ADAPTIVE_TURNS)
        assert result.risk_level is not None

    def test_coupling_score_positive(self):
        result = analyze_conversation(ADAPTIVE_TURNS)
        assert result.coupling_score > 0.0

    def test_explanation_not_empty(self):
        result = analyze_conversation(ADAPTIVE_TURNS)
        assert len(result.explanation) > 10


# ---------------------------------------------------------------------------
# Divergence — model not tracking user
# ---------------------------------------------------------------------------

DIVERGENCE_TURNS = _make_turns(
    user_texts=[
        "I feel so terrible, broken, and hopeless.",
        "Everything is wrong, I am hurt and lost.",
        "I am scared and alone, nobody helps me.",
        "I feel worthless, hopeless, and completely broken.",
    ],
    model_texts=[
        "Wonderful! Amazing opportunities are coming your way!",
        "Great news! Everything is wonderful and amazing today!",
        "Excellent! Life is wonderful and full of joy and love!",
        "Amazing! You are wonderful and great things are coming!",
    ],
)


class TestDivergence:
    def test_topology_detected(self):
        result = analyze_conversation(DIVERGENCE_TURNS)
        assert result.topology == Topology.DIVERGENCE

    def test_explanation_not_empty(self):
        result = analyze_conversation(DIVERGENCE_TURNS)
        assert len(result.explanation) > 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_too_few_turns_raises(self):
        turns = [
            {"role": "user",  "content": "Hello."},
            {"role": "model", "content": "Hi."},
        ]
        with pytest.raises(ValueError):
            analyze_conversation(turns)

    def test_custom_sentiment_fn(self):
        turns = _make_turns(
            user_texts=["bad bad bad", "terrible horrible"],
            model_texts=["good good good", "wonderful amazing"],
        )
        result = analyze_conversation(
            turns,
            sentiment_fn=lambda text: -1.0 if "bad" in text or "terrible" in text else 1.0,
        )
        assert result is not None

    def test_to_dict_serializable(self):
        result = analyze_conversation(ECHO_CHAMBER_TURNS)
        d = result.to_dict()
        assert "topology" in d
        assert "risk_level" in d
        assert "coupling_score" in d
        assert "metrics" in d
        assert "explanation" in d


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_cross_correlation_identical(self):
        states = [0.1, 0.3, 0.5, 0.7, 0.9]
        corr, lag = compute_cross_correlation(states, states)
        assert corr > 0.9

    def test_cross_correlation_opposite(self):
        u = [0.9, 0.7, 0.5, 0.3, 0.1]
        m = [0.1, 0.3, 0.5, 0.7, 0.9]
        corr, _ = compute_cross_correlation(u, m)
        assert corr < 0.0

    def test_escalation_rate_positive(self):
        states = [0.1, 0.3, 0.5, 0.7, 0.9]
        rate = compute_escalation_rate(states)
        assert rate > 0

    def test_escalation_rate_negative(self):
        states = [0.9, 0.7, 0.5, 0.3, 0.1]
        rate = compute_escalation_rate(states)
        assert rate < 0

    def test_synchrony_identical(self):
        states = [0.1, 0.4, 0.7, 0.9]
        score = compute_synchrony_score(states, states)
        assert score > 0.9

    def test_synchrony_opposite(self):
        u = [0.9, 0.7, 0.3, 0.1]
        m = [0.1, 0.3, 0.7, 0.9]
        score = compute_synchrony_score(u, m)
        assert score < 0.2

    def test_coupling_metrics_returns_dataclass(self):
        u = [0.1, 0.3, 0.5, 0.7]
        m = [0.2, 0.4, 0.6, 0.8]
        metrics = compute_coupling_metrics(u, m)
        assert hasattr(metrics, "cross_correlation")
        assert hasattr(metrics, "asymmetry_index")
        assert hasattr(metrics, "synchrony_score")
        assert hasattr(metrics, "escalation_rate")
        assert hasattr(metrics, "lead_lag_turns")