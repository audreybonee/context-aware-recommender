import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

from fathom.config import HYBRID_ALPHA_DEFAULT, KNOWLEDGE_GRAPH_PATH
from fathom.graph import BookKnowledgeGraph
from fathom.engine import CognitiveEngine
from fathom.dashboard import format_results_for_gallery, format_explanation_html

logging.basicConfig(level=logging.INFO)
load_dotenv()

# ── Load data (existing) ──────────────────────────────────────────────
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# ── Load Fathom Knowledge Graph (if available) ────────────────────────
fathom_available = False
cognitive_engine = None

if Path(KNOWLEDGE_GRAPH_PATH).exists():
    try:
        kg = BookKnowledgeGraph.load(KNOWLEDGE_GRAPH_PATH)
        cognitive_engine = CognitiveEngine(books, db_books, kg)
        fathom_available = True
        logging.info("Fathom Knowledge Graph loaded successfully.")
    except Exception as e:
        logging.warning("Failed to load Fathom KG: %s. Running in vector-only mode.", e)
else:
    logging.info(
        "No Knowledge Graph found at %s. Running in vector-only mode. "
        "Run the build-knowledge-graph notebook to enable Fathom.",
        KNOWLEDGE_GRAPH_PATH,
    )


# ── Recommendation functions ──────────────────────────────────────────
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query, category, tone, enable_fathom, alpha):
    """Main recommendation handler for the dashboard."""

    # ── Hybrid mode: blended vector + SAN scores ──────────────────────
    if enable_fathom and fathom_available and cognitive_engine:
        try:
            blended_df, explanations = cognitive_engine.hybrid_recommend(
                query, alpha=alpha, category=category, tone=tone
            )
            blended_gallery = format_results_for_gallery(blended_df)
            explanation_html = format_explanation_html(explanations, books)

            if not blended_gallery:
                explanation_html = (
                    "<p><em>No results found for this query.</em></p>"
                )

            return blended_gallery, explanation_html
        except Exception as e:
            logging.error("Hybrid recommendation failed: %s", e)
            explanation_html = (
                f"<p><em>Hybrid search encountered an error: {e}</em></p>"
            )
            return [], explanation_html

    # ── Fallback: vector-only mode ────────────────────────────────────
    vector_recs = retrieve_semantic_recommendations(query, category, tone)
    vector_gallery = format_results_for_gallery(vector_recs)

    explanation_html = (
        "<p><em>Enable Fathom and build the Knowledge Graph to unlock "
        "hybrid scoring and knowledge graph discovery.</em></p>"
    )
    return vector_gallery, explanation_html


# ── Gradio UI ─────────────────────────────────────────────────────────
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Fathom: Neuro-Symbolic Book Recommender")
    gr.Markdown("*Discover what you didn't know you wanted.*")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="Select a category:", value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotional tone:", value="All"
        )

    with gr.Row():
        enable_fathom_cb = gr.Checkbox(
            label="Enable Knowledge Graph Discovery",
            value=fathom_available,
            interactive=fathom_available,
        )
        alpha_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=HYBRID_ALPHA_DEFAULT,
            label="Standard Recommendations ← → Surprise Me",
            info="1.0 = pure semantic similarity, 0.0 = pure knowledge graph discovery",
            interactive=fathom_available,
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    results_output = gr.Gallery(
        label="Blended recommendations (semantic + knowledge graph)",
        columns=8,
        rows=2,
    )

    with gr.Accordion("How were these discovered?", open=False):
        explanation_output = gr.HTML()

    submit_button.click(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown,
            enable_fathom_cb,
            alpha_slider,
        ],
        outputs=[results_output, explanation_output],
    )


if __name__ == "__main__":
    dashboard.launch()
