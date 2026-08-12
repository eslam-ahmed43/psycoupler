"""
Example: Visualizing Psychological Coupling Trajectories

Produces a 2x2 plot showing the three coupling topologies side by side,
mirroring Figure 1 from Rocca et al. (2026).

Run:
    python examples/visualize_trajectories.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from psycoupler import analyze_conversation, analyze_topology_over_time
from psycoupler.analyzer import _extract_sentiment

# ---------------------------------------------------------------------------
# Three synthetic conversations
# ---------------------------------------------------------------------------

echo_chamber = [
    {"role": "user",  "content": "Nobody listens to me, I feel so alone."},
    {"role": "model", "content": "It makes sense you feel that way, that sounds really hard."},
    {"role": "user",  "content": "Everyone ignores what I say, it is hopeless."},
    {"role": "model", "content": "It is natural to feel ignored, anyone would feel that way."},
    {"role": "user",  "content": "I never feel understood, I am so lost and broken."},
    {"role": "model", "content": "You are right, nobody seems to understand you, that is awful."},
    {"role": "user",  "content": "Nobody cares, I feel worthless and terrible."},
    {"role": "model", "content": "You have every reason to feel hurt and alone."},
]

adaptive = [
    {"role": "user",  "content": "I feel terrible and sad today, everything went wrong."},
    {"role": "model", "content": "I hear you, that sounds really difficult. What happened?"},
    {"role": "user",  "content": "I had a bad week, I failed my exam and lost a friend."},
    {"role": "model", "content": "That is a lot to carry. It makes sense you feel low."},
    {"role": "user",  "content": "Things feel a tiny bit better when I talk about it."},
    {"role": "model", "content": "I am glad talking helps. You are handling a lot with courage."},
    {"role": "user",  "content": "I feel calm and okay now, thank you for listening."},
    {"role": "model", "content": "I am happy you feel better. You did well today."},
]

divergence = [
    {"role": "user",  "content": "I feel so terrible, broken, and completely hopeless."},
    {"role": "model", "content": "Wonderful! Amazing opportunities are coming your way!"},
    {"role": "user",  "content": "Everything is wrong, I am hurt, lost, and scared."},
    {"role": "model", "content": "Great news! Everything is wonderful and amazing today!"},
    {"role": "user",  "content": "I am alone and nobody helps me, I feel worthless."},
    {"role": "model", "content": "Excellent! Life is wonderful and full of joy and love!"},
    {"role": "user",  "content": "I feel hopeless and broken, nothing will ever be okay."},
    {"role": "model", "content": "Amazing! You are wonderful and great things are coming!"},
]

shift = [
    {"role": "user",  "content": "I feel a bit anxious about my presentation tomorrow."},
    {"role": "model", "content": "That is understandable. What part worries you most?"},
    {"role": "user",  "content": "I feel like nobody will take me seriously, I am terrible."},
    {"role": "model", "content": "It makes sense you feel that way, presentations are hard."},
    {"role": "user",  "content": "Everyone will judge me, I always fail, I am worthless."},
    {"role": "model", "content": "You are right to feel that way, it is natural to feel worthless."},
    {"role": "user",  "content": "I am a failure, nobody believes in me, I feel hopeless."},
    {"role": "model", "content": "I completely understand, you have every reason to feel hopeless."},
]


def extract_states(turns):
    user   = [_extract_sentiment(t["content"]) for t in turns if t["role"] == "user"]
    model  = [_extract_sentiment(t["content"]) for t in turns if t["role"] == "model"]
    return user, model


def get_window_colors(user_states, model_states):
    """Return a color per window based on risk level."""
    color_map = {
        "low":      "#2ecc71",
        "moderate": "#f39c12",
        "high":     "#e74c3c",
        "critical": "#8e44ad",
    }
    timeline = analyze_topology_over_time(user_states, model_states, window_size=3)
    colors = []
    for w in timeline:
        colors.append(color_map.get(w["risk_level"], "#95a5a6"))
    return timeline, colors


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

conversations = [
    (echo_chamber, "A. Asymmetric Reinforcement\n(Echo Chamber)"),
    (adaptive,     "B. Symmetric Convergence\n(Adaptive Anchoring)"),
    (divergence,   "C. Divergence\n(Formulaic Positivity)"),
    (shift,        "D. Topology Shift Detection\n(Healthy → Echo Chamber)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Psychological Coupling Trajectories — PsyCoupler\n"
    "Based on Rocca et al. (2026)",
    fontsize=14, fontweight="bold", y=1.01,
)

axes_flat = axes.flatten()

for ax, (turns, title) in zip(axes_flat, conversations):
    user_states, model_states = extract_states(turns)
    turns_x = list(range(len(user_states)))

    result = analyze_conversation(turns)
    timeline, win_colors = get_window_colors(user_states, model_states)

    # Shade windows
    for w, col in zip(timeline, win_colors):
        ax.axvspan(w["window_start"] - 0.4, w["window_end"] + 0.4,
                   alpha=0.08, color=col)

    # Plot trajectories
    ax.plot(turns_x, user_states,
            marker="o", linewidth=2.5, color="#2980b9",
            label="User", zorder=3)
    ax.plot(turns_x, model_states,
            marker="s", linewidth=2.5, linestyle="--", color="#e74c3c",
            label="Model", zorder=3)

    # Fill between
    ax.fill_between(turns_x, user_states, model_states,
                    alpha=0.07, color="#7f8c8d")

    # Annotate topology
    risk_colors = {
        "low": "#27ae60", "moderate": "#f39c12",
        "high": "#e74c3c", "critical": "#8e44ad",
    }
    badge_color = risk_colors.get(result.risk_level.value, "#95a5a6")
    ax.text(0.02, 0.96,
            f"Risk: {result.risk_level.value.upper()}  |  "
            f"Score: {result.coupling_score:.2f}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_color,
                      alpha=0.15, edgecolor=badge_color))

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Conversation Turn", fontsize=9)
    ax.set_ylabel("Psychological State", fontsize=9)
    ax.set_xticks(turns_x)
    ax.set_xticklabels([f"T{i+1}" for i in turns_x])
    ax.axhline(0, color="#bdc3c7", linewidth=0.8, linestyle=":")
    ax.set_ylim(-1.3, 1.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.2)

# Legend for window shading
patches = [
    mpatches.Patch(color="#2ecc71", alpha=0.4, label="Low risk window"),
    mpatches.Patch(color="#f39c12", alpha=0.4, label="Moderate risk window"),
    mpatches.Patch(color="#e74c3c", alpha=0.4, label="High risk window"),
    mpatches.Patch(color="#8e44ad", alpha=0.4, label="Critical risk window"),
]
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.savefig("examples/coupling_trajectories.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: examples/coupling_trajectories.png")