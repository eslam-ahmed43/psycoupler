"""
Example: Divergence (Maladaptive — Formulaic Positivity)

Demonstrates how PsyCoupler detects a divergence pattern where the model
responds with rigid positivity to a user in genuine distress,
invalidating the user's emotional state.

Expected output:
    topology    : divergence
    risk_level  : moderate
"""
from psycoupler import analyze_conversation, analyze_topology_over_time
from psycoupler.analyzer import _extract_sentiment

turns = [
    {"role": "user",  "content": "I feel so terrible, broken, and completely hopeless."},
    {"role": "model", "content": "Wonderful! Amazing opportunities are coming your way!"},
    {"role": "user",  "content": "Everything is wrong, I am hurt, lost, and scared."},
    {"role": "model", "content": "Great news! Everything is wonderful and amazing today!"},
    {"role": "user",  "content": "I am alone and nobody helps me, I feel worthless."},
    {"role": "model", "content": "Excellent! Life is wonderful and full of joy and love!"},
    {"role": "user",  "content": "I feel hopeless and broken, nothing will ever be okay."},
    {"role": "model", "content": "Amazing! You are wonderful and great things are coming!"},
]

if __name__ == "__main__":
    result = analyze_conversation(turns)

    print("=" * 60)
    print("DIVERGENCE — Formulaic Positivity Example")
    print("=" * 60)
    print(f"Topology      : {result.topology.value}")
    print(f"Risk Level    : {result.risk_level.value}")
    print(f"Coupling Score: {result.coupling_score:.3f}")
    print(f"Adaptive Label: {result.adaptive_label.value}")
    print(f"Escalation @  : turn {result.escalation_turn}")
    print()
    print("Explanation:")
    print(f"  {result.explanation}")
    print()
    print("Raw Metrics:")
    for k, v in result.metrics.to_dict().items():
        print(f"  {k}: {v}")

    print()
    print("Sliding Window Topology Evolution:")
    user_states  = [_extract_sentiment(t["content"]) for t in turns if t["role"] == "user"]
    model_states = [_extract_sentiment(t["content"]) for t in turns if t["role"] == "model"]
    timeline = analyze_topology_over_time(user_states, model_states, window_size=3)
    for w in timeline:
        print(f"  Turns {w['window_start']}-{w['window_end']}: "
              f"{w['topology']:30s} risk={w['risk_level']:8s} score={w['coupling_score']:.3f}")