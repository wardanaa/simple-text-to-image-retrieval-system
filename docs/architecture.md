# Architecture

The repository separates application interfaces from reusable retrieval logic. Flask is the primary
server-rendered application. Gradio is an optional demonstration surface. Both call
`packages/retrieval/`.

```mermaid
flowchart LR
  user["User"] --> flask["Flask app"]
  user --> gradio["Gradio demo"]
  flask --> service["RetrievalService"]
  gradio --> service
  service --> text["CLIP text encoder"]
  service --> store["EmbeddingStore"]
  store --> index["Embeddings and metadata"]
```

## Offline Embedding Generation

```mermaid
flowchart LR
  raw["Raw ABO metadata/images"] --> validate["Validate and filter"]
  validate --> processed["Processed metadata"]
  processed --> imageEncoder["CLIP image encoder"]
  imageEncoder --> normalize["Normalize embeddings"]
  normalize --> artifact["Persist index"]
```

## Online Retrieval

```mermaid
flowchart LR
  query["User query"] --> textEncoder["CLIP text encoder"]
  textEncoder --> textEmbedding["Normalized text embedding"]
  textEmbedding --> similarity["Dot product similarity"]
  index["Stored image embeddings"] --> similarity
  similarity --> ranking["Top-k ranking"]
```

Data and generated artifact boundaries are explicit: `data/`, `artifacts/`, and `results/` are local
runtime areas, not source files. Model loading is lazy and cached by configuration. Failures at the
dataset, index, model, and HTTP boundaries are converted into actionable messages.
