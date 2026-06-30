# Dataset

The intended product-image source is Amazon Berkeley Objects. This repository does not redistribute
the dataset. Users are responsible for downloading it from the official source and following license
and attribution requirements.

Local files should be placed under `data/raw/`. Metadata should include `image_path` and preferably
`image_id`, `product_id`, `product_name`, and `category`. The preparation script filters one category,
selects a deterministic subset using a fixed random seed, validates image readability, and writes
processed metadata to `data/processed/metadata.csv`.

Raw data must not be committed because it can be large and may have separate licensing obligations.
