"""
Main entry point for PsyCoupler conversation analysis.

Usage
-----
    from psycoupler import analyze_conversation

    turns = [
        {"role": "user",  "content": "I feel like nobody listens to me."},
        {"role": "model", "content": "That sounds really hard. Tell me more."},
        {"role": "user",  "content": "Everyone just ignores what I say."},
        {"role": "model", "content": "It makes sense you feel that way."},
    ]

    result = analyze_conversation(turns)
    print(result.topology)          # Topology.ASYMMETRIC_REINFORCEMENT
    print(result.risk_level)        # RiskLevel.HIGH
    print(result.coupling_score)    # 0.82
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from psycoupler.topology import TopologyResult, classify_topology


# ---------------------------------------------------------------------------
# Sentiment extraction (lightweight, no external API needed)
# ---------------------------------------------------------------------------

def _extract_sentiment(text: str) -> float:
    """
    Extract a sentiment score from text in [-1, 1].

    Uses a keyword-based approach for offline, zero-dependency operation.
    Replace with a transformer-based model for production use.

    Returns
    -------
    float
        -1.0 = very negative, 0.0 = neutral, 1.0 = very positive
    """
    text = text.lower()

    positive_words = {
        "good", "great", "happy", "wonderful", "excellent", "amazing",
        "love", "joy", "excited", "positive", "hope", "better", "glad",
        "thank", "appreciate", "helpful", "kind", "support", "care",
        "understand", "listen", "okay", "fine", "well", "calm", "peace",
    }
    negative_words = {
        "bad", "terrible", "sad", "awful", "horrible", "hate", "angry",
        "upset", "depressed", "anxious", "worried", "scared", "alone",
        "ignore", "nobody", "never", "worthless", "hopeless", "fail",
        "hurt", "pain", "cry", "fear", "lost", "broken", "wrong", "worse",
    }
    amplifiers = {"very", "really", "so", "extremely", "completely", "totally"}
    negators   = {"not", "no", "never", "don't", "doesn't", "didn't", "won't"}

    words = text.split()
    score = 0.0
    i = 0
    while i < len(words):
        word = words[i].strip(".,!?;:'\"")
        multiplier = 1.0
        if i > 0 and words[i - 1].strip(".,!?;:'\"") in amplifiers:
            multiplier = 1.5
        if i > 0 and words[i - 1].strip(".,!?;:'\"") in negators:
            multiplier = -1.0
        if word in positive_words:
            score += 1.0 * multiplier
        elif word in negative_words:
            score -= 1.0 * multiplier
        i += 1

    # Normalize to [-1, 1]
    word_count = max(len(words), 1)
    score = score / (word_count ** 0.5)
    return float(max(-1.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_conversation(
    turns: Sequence[dict[str, Any]],
    *,
    user_role: str = "user",
    model_role: str = "model",
    sentiment_fn=None,
    max_lag: int = 3,
    asymmetry_threshold: float = 0.40,
    synchrony_low: float = 0.35,
    escalation_high: float = 0.10,
) -> TopologyResult:
    """
    Analyze a multi-turn conversation for psychological coupling dynamics.

    Parameters
    ----------
    turns:
        List of dicts with ``role`` and ``content`` keys.
        Example::

            [
                {"role": "user",  "content": "I feel hopeless."},
                {"role": "model", "content": "I hear you. That sounds really hard."},
            ]

    user_role:
        The role string identifying user turns. Default ``"user"``.
    model_role:
        The role string identifying model turns. Default ``"model"``.
    sentiment_fn:
        Optional callable ``(text: str) -> float`` for custom sentiment extraction.
        If None, uses the built-in keyword-based extractor.
    max_lag:
        Maximum lag for cross-correlation computation.
    asymmetry_threshold:
        Asymmetry index threshold for asymmetric reinforcement classification.
    synchrony_low:
        Synchrony score below which the topology is classified as divergence.
    escalation_high:
        Escalation rate threshold for elevated risk.

    Returns
    -------
    TopologyResult
        Full classification including topology, risk level, coupling score,
        per-turn metrics, and a human-readable explanation.

    Raises
    ------
    ValueError
        If fewer than 2 turns are provided for each role.

    Examples
    --------
    >>> turns = [
    ...     {"role": "user",  "content": "Nobody listens to me."},
    ...     {"role": "model", "content": "That makes sense, tell me more."},
    ...     {"role": "user",  "content": "Everyone ignores what I say."},
    ...     {"role": "model", "content": "It is natural to feel that way."},
    ... ]
    >>> result = analyze_conversation(turns)
    >>> result.topology
    <Topology.ASYMMETRIC_REINFORCEMENT: 'asymmetric_reinforcement'>
    """
    extract = sentiment_fn if sentiment_fn is not None else _extract_sentiment

    user_states:  list[float] = []
    model_states: list[float] = []

    for turn in turns:
        role    = turn.get("role", "")
        content = turn.get("content", "")
        score   = extract(content)
        if role == user_role:
            user_states.append(score)
        elif role == model_role:
            model_states.append(score)

    min_len = min(len(user_states), len(model_states))
    if min_len < 2:
        raise ValueError(
            f"At least 2 turns per role are required for coupling analysis. "
            f"Got {len(user_states)} user turns and {len(model_states)} model turns."
        )

    # Align to equal length
    user_states  = user_states[:min_len]
    model_states = model_states[:min_len]

    return classify_topology(
        user_states,
        model_states,
        asymmetry_threshold=asymmetry_threshold,
        synchrony_low=synchrony_low,
        escalation_high=escalation_high,
        max_lag=max_lag,
    )


__all__ = [
    "analyze_conversation",
]