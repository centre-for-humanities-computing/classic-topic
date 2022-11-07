"""Module containing plotting utilities."""
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def topic_plot(genre_importance: pd.DataFrame, top_words: pd.DataFrame):
    """Plots genre and word importances for currently selected topic."""
    top_words = top_words.sort_values("relevance", ascending=False)
    genre_trace = go.Pie(
        values=genre_importance.importance,
        labels=genre_importance.group,
        textinfo="label",
        domain=dict(x=[0, 0.5]),
        showlegend=False,
        textposition="inside",
    )
    topic_word_trace = go.Bar(
        name="Importance for topic",
        y=top_words.word,
        x=top_words.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#dc2626",
    )
    overall_word_trace = go.Bar(
        name="Overall importance",
        y=top_words.word,
        x=top_words.overall_importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#f87171",
    )
    fig = make_subplots(
        specs=[
            [{"type": "domain"}, {"type": "xy"}],
        ],
        rows=1,
        cols=2,
        subplot_titles=("Most relevant genres", "Most relevant words"),
    )
    fig.add_trace(genre_trace, row=1, col=1)
    fig.add_trace(overall_word_trace, row=1, col=2)
    fig.add_trace(topic_word_trace, row=1, col=2)
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        barmode="overlay",
        plot_bgcolor="#f8fafc",
    )
    return fig


def all_topics_plot(topic_data: pd.DataFrame, current_topic: int) -> go.Figure:
    """Plots all topics on a bubble plot with estimated
    distances and importances.

    Parameters
    ----------
    topic_data: DataFrame
        Data about topic names, positions and sizes.

    Returns
    -------
    Figure
        Bubble plot of topics.
    """
    topic_data = topic_data.assign(
        selected=(topic_data.topic_id == current_topic).astype(int)
    )
    fig = px.scatter(
        topic_data,
        x="x",
        y="y",
        color="selected",
        text="topic_name",
        custom_data=["topic_id"],
        size="size",
        color_continuous_scale="Sunset_r",
    )
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        dragmode="pan",
    )
    fig.update_traces(textposition="top center", hovertemplate="")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(
        showticklabels=False,
        title="",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        linewidth=6,
        mirror=True,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    fig.update_yaxes(
        showticklabels=False,
        title="",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        mirror=True,
        linewidth=6,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    return fig


def documents_plot(document_data: pd.DataFrame) -> go.Figure:
    """Plots all documents in 3D space, colors them according to dominant topic.

    Parameters
    ----------
    document_data: DataFrame
        Data about document position, topic and metadata.

    Returns
    -------
    Figure
        3D Scatter plot of all documents.
    """
    fig = px.scatter_3d(
        document_data,
        x="x",
        y="y",
        z="z",
        color="topic_name",
        custom_data=[
            "værk",
            "forfatter",
            "group",
            "tlg_genre",
            "topic_name",
            "id_nummer",
        ],
    )
    # fig.update_traces(hovertemplate=None, hoverinfo="none")
    fig.update_traces(
        hovertemplate="""
            %{customdata[0]} - %{customdata[1]}<br>
            <i>Click to select</i>
        """
    )
    annotations = []
    for _index, row in document_data.iterrows():
        name = f"{row.værk} - {row.forfatter}"
        annotations.append(
            dict(
                x=row.x,
                y=row.y,
                z=row.z,
                text=name,
                bgcolor="white",
                bordercolor="black",
                arrowsize=1,
                arrowwidth=2,
                borderwidth=3,
                borderpad=10,
                font=dict(size=16, color="#0369a1"),
                visible=False,
                # clicktoshow="onout",
            )
        )
    axis = dict(
        showgrid=True,
        zeroline=True,
        visible=False,
    )
    fig.update_layout(
        # clickmode="event",
        # uirevision=True,
        modebar_remove=["lasso2d", "select2d"],
        hovermode="closest",
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
        hoverlabel=dict(font_size=11),
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis, annotations=annotations
        ),
    )
    return fig


def document_topic_plot(
    topic_importances: pd.DataFrame,
    topic_names: List[str],
) -> go.Figure:
    """Plots topic importances for a selected document.

    Parameters
    ----------
    topic_importances: dict of int to float
        Mapping of topic id's to importances.
    topic_names: list of str
        List of topic names.

    Returns
    -------
    Figure
        Pie chart of topic importances for each document.
    """
    name_mapping = pd.Series(topic_names)
    importances = topic_importances.assign(
        topic_name=topic_importances.topic_id.astype(int).map(name_mapping)
    )
    fig = px.pie(
        importances,
        values="importance",
        names="topic_name",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    print(importances)
    fig.update_traces(textposition="inside", textinfo="label")
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
    )
    return fig
