# Evaluation

Retrieval evaluation measures whether relevant items appear near the top of a ranked list. This differs
from classification, where each item is assigned a label.

Supported metrics:

- Precision@K: fraction of top-k results that are relevant.
- Recall@K: fraction of known relevant items recovered in top-k.
- Mean Reciprocal Rank: average inverse rank of the first relevant result.
- Mean Average Precision: ranking-sensitive precision across relevant hits.
- NDCG@K: ranking quality with discounted lower positions.

Metrics require relevance judgments such as relevant image IDs or product IDs. Synthetic examples are
useful for testing code paths but should not be treated as product-quality evidence.

```bash
python3 scripts/evaluate_retrieval.py --queries tests/fixtures/evaluation_queries.csv --k 5
```
