# Limitations and Future Work

## Current Limitations

- The default workflow targets a small single-category subset.
- CLIP may have bias and product-domain mismatch.
- The model is not fine-tuned on local product data.
- Product metadata quality affects display and evaluation.
- Similarity scores are not calibrated probabilities.
- Brute-force retrieval will not scale to large catalogs.
- Reliable evaluation requires relevance labels.

## Future Work

- Add FAISS or a vector database for larger catalogs.
- Support multilingual queries.
- Collect user feedback for relevance labeling.
- Explore CLIP fine-tuning or domain adaptation.
- Add richer product metadata and faceted filtering.
