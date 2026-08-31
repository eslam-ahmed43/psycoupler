"""
Dataset filtering for PsyCoupler.

Filters JSONL training datasets by coupling health score,
removing conversations with harmful psychological dynamics.
"""
import json
from typing import Any, Dict, List

from .analyzer import _extract_sentiment, analyze_conversation
from .topology import analyze_topology_over_time


def filter_dataset(
    input_path: str,
    output_path: str,
    max_risk: str = "moderate",
    batch_size: int = 100,
) -> None:
    """
    Filter a JSONL training dataset by coupling health.

    Conversations with harmful coupling dynamics above max_risk are removed.

    Parameters
    ----------
    input_path:
        Path to input JSONL file. Each line must be a JSON object with
        a ``conversations`` or ``turns`` key.
    output_path:
        Path to write filtered JSONL output.
    max_risk:
        Maximum allowed risk level: ``low``, ``moderate``, ``high``,
        ``critical``. Default ``moderate``.
    batch_size:
        Number of conversations per batch. Default 100.
    """
    risk_order = ["low", "moderate", "high", "critical"]
    max_index  = risk_order.index(max_risk.lower())

    with open(input_path,  "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        batch: List[Dict[str, Any]] = []
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError:
                continue

            if len(batch) >= batch_size:
                _process_batch(batch, outfile, max_index, risk_order)
                batch = []

        if batch:
            _process_batch(batch, outfile, max_index, risk_order)


def _process_batch(
    batch: List[Dict[str, Any]],
    outfile,
    max_index: int,
    risk_order: List[str],
) -> None:
    for conv in batch:
        turns = conv.get("conversations") or conv.get("turns")

        # No turns field — pass through unchanged
        if not turns:
            outfile.write(json.dumps(conv) + "\n")
            continue

        try:
            user_turns  = [t for t in turns if t.get("role") == "user"]
            model_turns = [t for t in turns if t.get("role") == "model"]

            # Too short to analyze — pass through
            if len(user_turns) < 2 or len(model_turns) < 2:
                outfile.write(json.dumps(conv) + "\n")
                continue

            # Primary check: full conversation analysis
            result  = analyze_conversation(turns)
            r_index = risk_order.index(result.risk_level.value.lower())
            dangerous = r_index > max_index

            # Flag maladaptive high-coupling conversations
            if not dangerous:
                if (
                    result.adaptive_label.value == "maladaptive"
                    and result.coupling_score > 0.50
                ):
                    dangerous = True

            # Secondary check: sliding window on longer conversations
            if not dangerous and len(user_turns) >= 5:
                user_states  = [_extract_sentiment(t["content"]) for t in user_turns]
                model_states = [_extract_sentiment(t["content"]) for t in model_turns]
                try:
                    timeline = analyze_topology_over_time(
                        user_states,
                        model_states,
                        window_size=5,
                        step=2,
                    )
                    for window in timeline:
                        w_index = risk_order.index(
                            window.get("risk_level", "low").lower()
                        )
                        if w_index > max_index or window.get("coupling_score", 0) > 0.75:
                            dangerous = True
                            break
                except Exception:
                    pass

            if not dangerous:
                outfile.write(json.dumps(conv) + "\n")

        except Exception as e:
            import sys
            print(f"Filter error: {e}", file=sys.stderr)
            outfile.write(json.dumps(conv) + "\n")