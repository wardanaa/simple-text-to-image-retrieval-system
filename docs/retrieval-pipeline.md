# Retrieval Pipeline

CLIP maps images and text into a shared embedding space. During indexing, each product image is
converted to RGB, processed by the CLIP processor, encoded by the image encoder, and L2-normalized.

During search, the query is validated, encoded by the CLIP text encoder, normalized, and compared with
all stored image embeddings. Because vectors are normalized, cosine similarity is a dot product.

The system sorts scores in descending order and returns the top-k rows. Scores are useful for ranking
within a query but are not calibrated probabilities. Brute-force search is acceptable for the initial
500 to 1,000 image subset because it is simple and transparent for students.
