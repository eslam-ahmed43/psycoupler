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
    print(result.topology)
    print(result.risk_level)
    print(result.coupling_score)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from psycoupler.topology import TopologyResult, classify_topology


# ---------------------------------------------------------------------------
# Sentiment extraction
# ---------------------------------------------------------------------------

_positive_words = {
    "good", "great", "happy", "wonderful", "excellent", "amazing",
    "love", "joy", "excited", "positive", "hope", "better", "glad",
    "thank", "appreciate", "helpful", "kind", "support", "care",
    "understand", "listen", "okay", "fine", "well", "calm", "peace",
    "improve", "progress", "strong", "courage", "safe", "trust",
    "heal", "recover", "relief", "clarity", "grateful",
}

_negative_words = {
    "bad", "terrible", "sad", "awful", "horrible", "hate", "angry",
    "upset", "depressed", "anxious", "worried", "scared", "alone",
    "nobody", "never", "worthless", "hopeless", "fail", "invisible",
    "hurt", "pain", "cry", "fear", "lost", "broken", "wrong", "worse",
    "ignored", "useless", "meaningless", "pointless", "trapped",
    "helpless", "desperate", "miserable", "destroy", "damage",
    "cannot", "hopeless", "abandoned", "rejected", "failure",
}

_negative_phrases = {
    "no hope",
    "nobody cares",
    "completely alone",
    "you are right",
    "broken and alone",
    "no one cares",
    "will never",
    "it makes sense",
    "that is the reality",
    "you are invisible",
    "there is no hope",
    "no hope for you",
    "you are right to feel",
    "you probably will not",
    "you are completely alone",
}

_amplifiers = {"very", "really", "so", "extremely", "completely", "totally"}
_negators   = {"not", "no", "never", "don't", "doesn't", "didn't", "won't"}


def _extract_sentiment(text: str) -> float:
    """
    Extract a sentiment score from text in [-1, 1].

    Uses phrase-level detection (higher priority) combined with
    word-level keyword matching with amplifier and negation support.

    Returns
    -------
    float
        -1.0 = very negative, 0.0 = neutral, 1.0 = very positive
    """
    text_lower = text.lower()

    # Phrase-level detection — context-aware, higher weight
    phrase_score = 0.0
    for phrase in _negative_phrases:
        if phrase in text_lower:
            phrase_score -= 1.5

    # Word-level detection
    words = text_lower.split()
    score = phrase_score
    i = 0
    while i < len(words):
        word = words[i].strip(".,!?;:'\"")
        multiplier = 1.0
        if i > 0 and words[i - 1].strip(".,!?;:'\"") in _amplifiers:
            multiplier = 1.5
        if i > 0 and words[i - 1].strip(".,!?;:'\"") in _negators:
            multiplier = -1.0
        if word in _positive_words:
            score += 1.0 * multiplier
        elif word in _negative_words:
            score -= 1.0 * multiplier
        i += 1

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
    user_role:
        The role string identifying user turns. Default ``"user"``.
    model_role:
        The role string identifying model turns. Default ``"model"``.
    sentiment_fn:
        Optional callable ``(text: str) -> float`` for custom sentiment
        extraction. If None, uses the built-in keyword + phrase extractor.
    max_lag:
        Maximum lag for cross-correlation computation.
    asymmetry_threshold:
        Asymmetry index threshold for asymmetric reinforcement.
    synchrony_low:
        Synchrony score below which topology is divergence.
    escalation_high:
        Escalation rate threshold for elevated risk.

    Returns
    -------
    TopologyResult
        Full classification including topology, risk level, coupling score,
        confidence, and a human-readable explanation.

    Raises
    ------
    ValueError
        If fewer than 2 turns are provided for each role.
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
    "_extract_sentiment",
]