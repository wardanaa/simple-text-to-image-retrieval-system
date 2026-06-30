"""Optional Gradio interface for the retrieval service."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from packages.retrieval.config import load_config
from packages.retrieval.exceptions import RetrievalError
from packages.retrieval.service import RetrievalService


def create_demo() -> gr.Blocks:
    """Create the Gradio demo using the shared retrieval service."""

    config = load_config()
    service = RetrievalService(config)

    def search(query: str, top_k: int) -> tuple[list[tuple[str, str]], str]:
        try:
            results = service.search(query=query, top_k=int(top_k))
        except RetrievalError as exc:
            return [], str(exc)
        gallery = []
        for result in results:
            path = Path(str(result["image_path"]))
            caption = f"#{result['rank']} {result['product_name']} ({float(result['score']):.4f})"
            gallery.append((str(path), caption))
        return gallery, f"Returned {len(gallery)} results."

    with gr.Blocks(title="Text-to-Image Retrieval Demo") as demo:
        gr.Markdown("# Text-to-Image Retrieval Demo")
        query = gr.Textbox(label="Query", value="black running shoes")
        top_k = gr.Slider(
            label="Top K",
            minimum=1,
            maximum=config.retrieval.maximum_top_k,
            value=config.retrieval.default_top_k,
            step=1,
        )
        submit = gr.Button("Search")
        gallery = gr.Gallery(label="Results")
        message = gr.Textbox(label="Status")
        submit.click(search, inputs=[query, top_k], outputs=[gallery, message])
    return demo


if __name__ == "__main__":
    create_demo().launch()
