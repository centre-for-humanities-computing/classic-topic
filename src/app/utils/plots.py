import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def topic_plot(
    topic: int, genre_importance: pd.DataFrame, top_words: pd.DataFrame
):
    """Plots genre and word importances for currently selected topic.

    Parameters
    ----------
    topic: int
        Index of the topic to be displayed.
    genre_importance: DataFrame
        Data about genre importances.
    top_words: DataFrame
        Data about word importances for each topic.

    Returns
    -------
    Figure
        Piechart of genre importances and bar chart of word importances.
    """
    genre_importance = genre_importance[genre_importance.topic == topic]
    top_words = top_words[top_words.topic == topic]
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
    """Plots all topics on a bubble plot with estimated distances and importances.

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
    fig = px.scatter_3d(
        document_data,
        x="x",
        y="y",
        z="z",
        color="topic_name",
        custom_data=["værk", "forfatter", "group", "tlg_genre", "topic_name"],
    )
    fig.update_traces(
        hovertemplate="""
            <b>%{customdata[0]} - %{customdata[1]}</b><br>
            Dominant topic: <i> %{customdata[4]} </i> <br>
            TLG genre: <i> %{customdata[3]} </i> <br>
            Group: <i> %{customdata[2]} </i> <br>
            <br>
            <i>Click for more information...</i>
        """
    )
    axis = dict(
        showgrid=True,
        zeroline=True,
        visible=False,
    )
    fig.update_layout(
        clickmode="event",
        modebar_remove=["lasso2d", "select2d"],
        hovermode="closest",
        paper_bgcolor="rgba(1,1,1,0)",
        plot_bgcolor="rgba(1,1,1,0)",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            # font_family="Rockwell"
        ),
    )
    return fig
