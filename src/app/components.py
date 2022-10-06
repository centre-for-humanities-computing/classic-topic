# type: ignore
from dash import html, dcc

from app import styles


def Label(text="", size=15, text_align="left", padding_top=20, **kwargs):
    return html.Label(
        text,
        style={
            "padding-top": f"{padding_top}px",
            "margin-bottom": "7px",
            "font": f"{size}px Helvetica",
            "text-align": text_align,
            **kwargs,
        },
    )


def Spacer(px=None):
    if px is None:
        return html.Div(style={"flex-grow": "1"})
    else:
        return html.Div(style={"height": f"{px}px"})


sidebar_body = html.Div(
    id="sidebar_body",
    style=styles.sidebar_body,
    children=[
        Label("Pipeline settings ⚙️", size=25, text_align="left"),
        Label("Vectorizer options", size=20, padding_top=30),
        Label("Vectorizer", size=15, padding_top=10),
        dcc.Dropdown(
            id="select_vectorizer",
            options={
                "bow": "Bag of Words",
                "tf_idf": "Term Frequency-Inverse Document Frequency",
            },
            value="bow",
        ),
        Label("Maximum word document frequency", size=15),
        dcc.Slider(
            id="max_df",
            min=0.0,
            max=1.0,
            step=0.1,
            value=0.1,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        Label("Minimum word document occurrance", size=15),
        dcc.Slider(
            id="min_df",
            min=0,
            max=50,
            value=10,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        Label("Model options", size=20, padding_top=30),
        Label("Topic model", size=15, padding_top=10),
        dcc.Dropdown(
            id="select_model",
            options={
                "nmf": "Non-negative Matrix Factorization",
                "lda": "Latent Dirichlet Allocation",
                "lsa": "Latent Semantic Allocation/Indexing",
                "dmm": "Dirichlet Multinomial Mixture (best for short texts)",
            },
            value="nmf",
        ),
        Label("Number of topics", size=15),
        dcc.Slider(
            id="n_topics",
            min=10,
            max=100,
            step=10,
            value=100,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        Spacer(),
        html.Button(
            "Fit pipeline ✔️",
            id="fit_pipeline",
            n_clicks=0,
            style=styles.submit,
        ),
    ],
)

sidebar_collapser = html.Button(
    ">", id="sidebar_collapser", style=styles.collapse_button, n_clicks=0
)

topic_switcher = html.Div(
    id="topic_switcher",
    style=styles.topic_switcher,
    children=[
        html.Button(
            "<- Previous topic",
            id="prev_topic",
            style=styles.next_prev_topic,
            n_clicks=0,
        ),
        html.Div(
            [
                html.Div(
                    "Rename current topic: ",
                    style={
                        "color": "#555555",
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "font": "18px Helvetica",
                    },
                ),
                dcc.Input(
                    id="topic_name",
                    type="text",
                    value="Topic 1",
                    style=styles.topic_namer,
                    debounce=True,
                ),
            ],
            style={
                "display": "flex",
                "flex": "4 0",
                "justify-content": "center",
                "align-items": "stretch",
                "border": "none",
            },
        ),
        html.Button(
            "Next topic ->",
            id="next_topic",
            style=styles.next_prev_topic,
            n_clicks=0,
        ),
    ],
)
