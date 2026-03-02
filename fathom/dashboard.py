from typing import List, Tuple

import pandas as pd


def format_book_card(row: pd.Series) -> Tuple[str, str]:
    """Format a book row as (thumbnail_url, caption) for Gradio Gallery."""
    description = str(row.get("description", ""))
    truncated = " ".join(description.split()[:30]) + "..."

    authors_raw = str(row.get("authors", "Unknown"))
    authors_split = authors_raw.split(";")
    if len(authors_split) == 2:
        authors_str = f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    else:
        authors_str = authors_raw

    caption = f"{row['title']} by {authors_str}: {truncated}"
    return (row.get("large_thumbnail", "cover-not-found.jpg"), caption)


def format_results_for_gallery(
    results_df: pd.DataFrame,
) -> List[Tuple[str, str]]:
    """Convert a results DataFrame into Gradio Gallery format."""
    gallery_items = []
    for _, row in results_df.iterrows():
        gallery_items.append(format_book_card(row))
    return gallery_items


def format_explanation_html(
    explanations: List[dict],
    books_df: pd.DataFrame,
) -> str:
    """Build HTML showing the KG path for each SAN-discovered book.

    Each explanation shows:
      "Book Title" discovered via: concept1, concept2 (shared with "Seed Title")
    """
    if not explanations:
        return "<p><em>No knowledge graph connections found.</em></p>"

    # Build isbn → title lookup
    isbn_to_title = dict(
        zip(books_df["isbn13"].astype(str), books_df["title"])
    )

    html_parts = ['<div style="font-size: 14px; line-height: 1.8;">']

    for exp in explanations:
        target_title = exp.get("title", "Unknown")
        seed_title = isbn_to_title.get(exp["seed_isbn"], "Unknown")
        via = exp.get("via_concepts", [])

        # Format concept names for display (remove prefixes, replace hyphens)
        concept_labels = []
        for node in via:
            # node format: "concept:redemption" or "author:name"
            parts = node.split(":", 1)
            label = parts[1] if len(parts) > 1 else parts[0]
            label = label.replace("-", " ").title()
            node_type = parts[0] if len(parts) > 1 else "concept"
            concept_labels.append(f'<strong>{label}</strong> ({node_type})')

        via_str = " &rarr; ".join(concept_labels) if concept_labels else "direct"

        html_parts.append(
            f'<p>&bull; <em>"{target_title}"</em> discovered via: '
            f'{via_str} &mdash; shared with <em>"{seed_title}"</em></p>'
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)
