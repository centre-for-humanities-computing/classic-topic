from typing import Dict, List

from dash import dcc, html

from app.utils.plots import document_topic_plot


def document_tooltip(
    work: str,
    author: str,
):
    """Constructs document tooltip based on document data.

    Parameters
    ----------
    work: str
        Name of the document.
    author: str
        Author of the document.

    Returns
    -------
    list of dash component
        Children of the tooltip.
    """
    if len(work) > 35:
        work = work[:30] + "..."
    if author and len(author) > 35:
        author = author[:30] + "..."
    return [
        html.Div(
            [
                html.H1(f"{work}", className="text-xl break-normal"),
                html.H2(
                    f"{author}",
                    className=" break-normal italic text-gray-700 text-lg",
                ),
            ],
            className="flex flex-col p-5",
        )
    ]
