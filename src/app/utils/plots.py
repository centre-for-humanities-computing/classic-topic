import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def genre_plot(topic: int, importance: pd.DataFrame) -> go.Pie:
    """Creates a piechart trace visualizing the relevance of
    different genres for a topic.

    Parameters
    ----------
    topic: int
        Index of the topic
    importance: DataFrame
        Table containing information about the
        importance of topics for each group.

    Returns
    -------
    Pie
        Trace of the piechart.
    """
    importance = importance[importance.topic == topic]
    return go.Pie(
        values=importance.importance,
        labels=importance.group,
        textinfo="label",
        domain=dict(x=[0, 0.5]),
        showlegend=False,
    )


def word_plot(topic: int, top_words: pd.DataFrame) -> go.Bar:
    """Shows top words for a topic on a horizontal bar plot.

    Parameters
    ----------
    topic: int
        Index of the topic
    top_words: DataFrame
        Table containing information about word importances
        for each topic.

    Returns
    -------
    Bar
        Bar chart visualizing the top words for a topic.
    """
    vis_df = top_words[top_words.topic == topic]
    return go.Bar(
        y=vis_df.word,
        x=vis_df.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        showlegend=False,
    )


def join_plots(row: pd.Series) -> go.Figure:
    """Joins the plots together in one row of the data frame.

    Parameters
    ----------
    row: Series
        Series representing one row of a dataframe containing
        the index of the topic, a bar chart and a pie chart.

    Returns
    -------
    Figure
        Joint plot of the pie and bar charts with titles added.
    """
    fig = make_subplots(
        specs=[
            [{"type": "domain"}, {"type": "xy"}],
        ],
        rows=1,
        cols=2,
        subplot_titles=("Most relevant genres", "Most relevant words"),
    )
    fig.add_trace(row.genre_fig, row=1, col=1)
    fig.add_trace(row.word_fig, row=1, col=2)
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=900,
        title_text=f"Topic {row.topic}",
    )
    return fig
