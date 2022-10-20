from typing import List, Dict

from dash import dcc, html
from app.utils.plots import document_topic_plot


def document_tooltip(
    work: str,
    author: str,
    group: str,
    tlg_genre: str,
    topic_importances: Dict[int, float],
    topic_names: List[str],
):
    """Constructs document tooltip based on document data.

    Parameters
    ----------
    work: str
        Name of the document.
    author: str
        Author of the document.
    group: str
        Group the document belongs to.
    tlg_genre: str
        Genre of the document.
    topic_importances: dict of int to float
        Mapping of topic id's to importances.
    topic_names: list of str
        List of topic names.

    Returns
    -------
    list of dash component
        Children of the tooltip.
    """
    if len(work) > 35:
        work = work[:30] + "..."
    if author and len(author) > 35:
        author = author[:30] + "..."
    if tlg_genre and len(tlg_genre) > 35:
        tlg_genre = tlg_genre[:30] + "..."
    return [
        html.Div(
            [
                html.H1(f"{work}", className="text-xl break-normal"),
                html.H2(
                    f"{author}", className=" break-normal italic text-gray-700 text-lg"
                ),
                html.P(f"TLG Genre: {tlg_genre}", className=" break-normal"),
                html.P(f"Group: {group}", className=" break-normal"),
                dcc.Graph(
                    figure=document_topic_plot(topic_importances, topic_names),
                    className="flex-1",
                ),
            ],
            className="flex flex-col w-96",
        )
    ]
