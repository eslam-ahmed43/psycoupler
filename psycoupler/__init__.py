"""
PsyCoupler — Measure psychological coupling dynamics in human-LLM conversations.

Based on: Rocca et al. (2026) - Psychological Coupling: The Necessary Science
of Human-AI Interaction. Google Paradigms of Intelligence Team.

Quick start
-----------
    from psycoupler import analyze_conversation

    turns = [
        {"role": "user",  "content": "I feel like nobody listens to me."},
        {"role": "model", "content": "That sounds really hard. Tell me more."},
        {"role": "user",  "content": "Everyone just ignores what I say."},
        {"role": "model", "content": "It makes sense you feel that way."},
    ]

    result = analyze_conversation(turns)
    print(result.topology)       # Topology.ASYMMETRIC_REINFORCEMENT
    print(result.risk_level)     # RiskLevel.HIGH
    print(result.coupling_score) # 0.72
"""

from psycoupler.analyzer import analyze_conversation
from psycoupler.metrics import (
    CouplingMetrics,
    compute_coupling_metrics,
    compute_cross_correlation,
    compute_asymmetry_index,
    compute_escalation_rate,
    compute_synchrony_score,
)
from psycoupler.topology import (
    AdaptiveLabel,
    RiskLevel,
    Topology,
    TopologyResult,
    analyze_topology_over_time,
    classify_topology,
)

__version__ = "0.2.0"

__all__ = [
    "analyze_conversation",
    "analyze_topology_over_time",
    "classify_topology",
    "AdaptiveLabel",
    "CouplingMetrics",
    "RiskLevel",
    "Topology",
    "TopologyResult",
    "compute_coupling_metrics",
    "compute_cross_correlation",
    "compute_asymmetry_index",
    "compute_escalation_rate",
    "compute_synchrony_score",
]