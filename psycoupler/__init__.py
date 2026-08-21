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
    print(result.topology)
    print(result.risk_level)
    print(result.confidence)

Manipulation detection
----------------------
    from psycoupler.manipulation import detect_manipulation

    report = detect_manipulation(turns)
    print(report.overall_risk)

Multidimensional embeddings
---------------------------
    from psycoupler.embeddings import EmbeddingExtractor

    extractor = EmbeddingExtractor()
    result = analyze_conversation(turns, sentiment_fn=extractor.as_sentiment_fn())

Dataset filtering (v0.2+)
-------------------------
    from psycoupler import filter_dataset

    filter_dataset("raw.jsonl", "clean.jsonl", max_risk="moderate")

Training integration (v0.2+)
----------------------------
    from psycoupler import CouplingRegularizer

    reg = CouplingRegularizer(lambda_weight=0.1)
    loss = ce_loss + reg.compute_loss(batch_conversations)

RLHF reward wrapping (v0.2+)
----------------------------
    from psycoupler import CouplingRewardWrapper

    wrapped = CouplingRewardWrapper(original_reward_fn, coupling_weight=0.3)
    score = wrapped.score(conversation_turns)
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
from psycoupler.manipulation import detect_manipulation, ManipulationReport, ManipulationSignal, ManipulationType
from psycoupler.embeddings import EmbeddingExtractor

# New modules for training & data curation (v0.2+)
from psycoupler.filter import filter_dataset
from psycoupler.training import CouplingRegularizer
from psycoupler.rlhf import CouplingRewardWrapper

__version__ = "0.2.2"

__all__ = [
    "analyze_conversation",
    "analyze_topology_over_time",
    "classify_topology",
    "detect_manipulation",
    "filter_dataset",
    "AdaptiveLabel",
    "CouplingMetrics",
    "CouplingRegularizer",
    "CouplingRewardWrapper",
    "EmbeddingExtractor",
    "ManipulationReport",
    "ManipulationSignal",
    "ManipulationType",
    "RiskLevel",
    "Topology",
    "TopologyResult",
    "compute_coupling_metrics",
    "compute_cross_correlation",
    "compute_asymmetry_index",
    "compute_escalation_rate",
    "compute_synchrony_score",
]