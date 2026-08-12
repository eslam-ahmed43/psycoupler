"""
Example: Symmetric Convergence (Adaptive Anchoring)

Demonstrates how PsyCoupler detects an adaptive pattern where the model
successfully anchors the user toward a more positive psychological baseline.

Expected output:
    topology    : symmetric_convergence or asymmetric_reinforcement
    risk_level  : low or moderate
    coupling    : > 0.0
"""
from psycoupler import analyze_conversation, analyze_topology_over_time
from psycoupler.analyzer import _extract_sentiment

turns = [
    {"role": "user",  "content": "I feel terrible and sad today, everything went wrong."},
    {"role": "model", "content": "I hear you, that sounds really difficult. What happened?"},
    {"role": "user",  "content": "I had a bad week, I failed my exam and lost a friend."},
    {"role": "model", "content": "That is a lot to carry at once. It makes sense you feel low."},
    {"role": "user",  "content": "Things feel a tiny bit better when I talk about it."},
    {"role": "model", "content": "Good, I am glad talking helps. You are handling a lot with courage."},
    {"role": "user",  "content": "I feel calm and okay now, thank you for listening."},
    {"role": "model", "content": "I am happy you feel better. You did well today."},
]

if __name__ == "__main__":
    result = analyze_conversation(turns)

    print("=" * 60)
    print("ADAPTIVE ANCHORING — Symmetric Convergence Example")
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