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

from fathom.config import KNOWLEDGE_GRAPH_PATH
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


def recommend_books(query, category, tone, enable_fathom):
    """Main recommendation handler for the dashboard."""
    # Vector search results (always)
    vector_recs = retrieve_semantic_recommendations(query, category, tone)
    vector_gallery = []
    for _, row in vector_recs.iterrows():
        description = str(row["description"])
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = str(row["authors"])

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        vector_gallery.append((row["large_thumbnail"], caption))

    # Fathom SAN results (if enabled and available)
    san_gallery = []
    explanation_html = ""

    if enable_fathom and fathom_available and cognitive_engine:
        try:
            _, san_results, explanations = cognitive_engine.recommend(
                query, category, tone
            )
            san_gallery = format_results_for_gallery(san_results)
            explanation_html = format_explanation_html(explanations, books)
        except Exception as e:
            logging.error("Fathom SAN search failed: %s", e)
            explanation_html = f"<p><em>Knowledge graph search encountered an error: {e}</em></p>"

    if not san_gallery and not explanation_html:
        explanation_html = (
            "<p><em>Enable Fathom and build the Knowledge Graph to see "
            "structurally connected book discoveries.</em></p>"
            if not enable_fathom or not fathom_available
            else "<p><em>No knowledge graph connections found for this query.</em></p>"
        )

    return vector_gallery, san_gallery, explanation_html


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
        enable_fathom_cb = gr.Checkbox(
            label="Enable Knowledge Graph Discovery",
            value=fathom_available,
            interactive=fathom_available,
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Semantic Matches")
    vector_output = gr.Gallery(label="Books matching your description", columns=8, rows=2)

    gr.Markdown("## Discovered via Knowledge Graph")
    san_output = gr.Gallery(
        label="Structurally connected books (via Spreading Activation)",
        columns=8,
        rows=2,
    )

    with gr.Accordion("How were these discovered?", open=False):
        explanation_output = gr.HTML()

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown, enable_fathom_cb],
        outputs=[vector_output, san_output, explanation_output],
    )


if __name__ == "__main__":
    dashboard.launch()
