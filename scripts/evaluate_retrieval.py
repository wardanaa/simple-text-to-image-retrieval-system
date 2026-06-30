"""Evaluate retrieval results with relevance labels when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from packages.retrieval.config import load_config
from packages.retrieval.service import RetrievalService


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Calculate Precision@K."""

    if k < 1:
        raise ValueError("k must be positive.")
    return sum(item in relevant for item in retrieved[:k]) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Calculate Recall@K when relevant labels exist."""

    if not relevant:
        return None
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Calculate reciprocal rank."""

    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Calculate average precision up to K."""

    if not relevant:
        return None
    hits = 0
    total = 0.0
    for index, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / index
    return total / min(len(relevant), k)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Calculate binary NDCG@K."""

    if not relevant:
        return None
    gains = np.array([1.0 if item in relevant else 0.0 for item in retrieved[:k]])
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_hits = min(len(relevant), k)
    ideal_gains = np.ones(ideal_hits)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    return dcg / idcg if idcg else None


def parse_relevant(value: object) -> set[str]:
    """Parse pipe- or comma-separated relevance IDs."""

    if pd.isna(value):
        return set()
    text = str(value)
    separator = "|" if "|" in text else ","
    return {part.strip() for part in text.split(separator) if part.strip()}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("tests/fixtures/evaluation_queries.csv")
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    return parser.parse_args()


def main() -> int:
    """Run retrieval evaluation from a labeled query file."""

    args = parse_args()
    queries = pd.read_csv(args.queries)
    required = {"query", "relevant_image_ids"}
    if not required.issubset(queries.columns):
        raise ValueError("Evaluation queries must include query and relevant_image_ids columns.")
    service = RetrievalService(load_config())
    rows: list[dict[str, object]] = []
    for _, query_row in queries.iterrows():
        results = service.search(str(query_row["query"]), top_k=args.k)
        retrieved = [str(result["image_id"]) for result in results]
        relevant = parse_relevant(query_row["relevant_image_ids"])
        rows.append(
            {
                "query": query_row["query"],
                "precision_at_k": precision_at_k(retrieved, relevant, args.k),
                "recall_at_k": recall_at_k(retrieved, relevant, args.k),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "average_precision": average_precision(retrieved, relevant, args.k),
                "ndcg_at_k": ndcg_at_k(retrieved, relevant, args.k),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(pd.DataFrame(rows).describe(include="all").to_string())
    print(f"Wrote evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
