import json
from typing import List, Dict, Any

from .analyzer import _extract_sentiment
from .topology import analyze_topology_over_time

def filter_dataset(
    input_path: str,
    output_path: str,
    max_risk: str = "moderate",
    batch_size: int = 100
) -> None:
    risk_order = ["low", "moderate", "high", "critical"]
    max_index = risk_order.index(max_risk.lower())

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        batch = []
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                batch.append(data)
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
    risk_order: List[str]
) -> None:
    for conv in batch:
        turns = conv.get("conversations") or conv.get("turns")
        if not turns:
            outfile.write(json.dumps(conv) + '\n')
            continue

        try:
            user_turns = [t for t in turns if t.get("role") == "user"]
            model_turns = [t for t in turns if t.get("role") == "model"]

            if len(user_turns) < 3 or len(model_turns) < 3:
                outfile.write(json.dumps(conv) + '\n')
                continue

            user_states = [_extract_sentiment(t["content"]) for t in user_turns]
            model_states = [_extract_sentiment(t["content"]) for t in model_turns]

            timeline = analyze_topology_over_time(
                user_states,
                model_states,
                window_size=5,
                step=2
            )

            dangerous = False
            for window in timeline:
                risk = window.get("risk_level", "low").lower()
                score = window.get("coupling_score", 0.0)

                if risk in ["high", "critical"] or score > 0.65:
                    dangerous = True
                    break

            if not dangerous:
                outfile.write(json.dumps(conv) + '\n')

        except Exception as e:
            import sys
            print(f"Filter error: {e}", file=sys.stderr)
            outfile.write(json.dumps(conv) + '\n')