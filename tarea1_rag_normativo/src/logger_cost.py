"""Registro de costos de LLM en logs/cost_log.csv (una fila por consulta)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

_FIELDNAMES = [
    "timestamp_utc", "pregunta", "abstained", "tokens_in", "tokens_out",
    "cost_usd", "latency_sec",
]


def compute_cost_usd(tokens_in: int, tokens_out: int, price_input_per_1k: float, price_output_per_1k: float) -> float:
    return (tokens_in / 1000) * price_input_per_1k + (tokens_out / 1000) * price_output_per_1k


def log_query_cost(cost_log_path: Path, pregunta: str, abstained: bool, tokens_in: int, tokens_out: int,
                    cost_usd: float, latency_sec: float) -> None:
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not cost_log_path.exists()
    with cost_log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pregunta": pregunta,
            "abstained": abstained,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 6),
            "latency_sec": round(latency_sec, 3),
        })
