# type: ignore
from dash import dcc, html


def hamburger_icon(color: str) -> html.Div:
    return html.Div(
        className="space-y-2",
        children=[
            html.Span(className=f"block w-6 h-0.5 {color} rounded-full"),
        ]
        * 3,
    )


sidebar_body_class = """
    flex fixed w-3/12 h-full flex-col p-8 bg-white shadow-lg rounded-xl
    transition-all ease-in space-y-3 top-0 right-0
"""

sidebar = html.Div(
    id="sidebar",
    children=[
        html.Div(
            className="""
            fixed flex-0 flex-row justify-end content-around
            top-0 right-0 z-20 m-3 mr-4
            """,
            children=html.Button(
                hamburger_icon("bg-sky-700"),
                id="sidebar_collapser",
                n_clicks=0,
                className="""
                    flex justify-center content-center
                    w-14 flex-0 w-12 h-12 m-1 mt-8
                """,
            ),
        ),
        html.Div(
            id="sidebar_body",
            className=sidebar_body_class + " translate-x-0",
            children=[
                html.H1(
                    "Pipeline Settings ⚙️",
                    className="text-2xl mt-2 mb-3",
                ),
                html.Span(
                    className="""
                    block w-full bg-sky-50 h-0.5
                    """
                ),
                html.Span(className="h-5"),
                html.H2("Vectorizer options", className="text-xl mb-2"),
                html.H3("Vectorizer", className="text-base mb-2"),
                dcc.Dropdown(
                    id="select_vectorizer",
                    className="mb-2",
                    options={
                        "bow": "Bag of Words",
                        "tf_idf": "Term Frequency-Inverse Document Frequency",
                    },
                    value="bow",
                ),
                html.H3(
                    "Maximum word document frequency",
                    className="text-base mb-2",
                ),
                dcc.Slider(
                    id="max_df",
                    className="mb-2",
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    value=0.1,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H3(
                    "Minimum word document occurrance",
                    className="text-base mb-2",
                ),
                dcc.Slider(
                    id="min_df",
                    className="mb-2",
                    min=0,
                    max=50,
                    value=10,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H2("Model options", className="text-xl mb-2"),
                html.H3("Topic model", className="text-base mb-2"),
                dcc.Dropdown(
                    id="select_model",
                    className="mb-2",
                    options={
                        "nmf": "Non-negative Matrix Factorization",
                        "lda": "Latent Dirichlet Allocation",
                        "lsa": "Latent Semantic Allocation/Indexing",
                        "dmm": "Dirichlet Multinomial Mixture (best for short texts)",
                    },
                    value="nmf",
                ),
                html.H3("Number of topics", className="text-base mb-2"),
                dcc.Slider(
                    id="n_topics",
                    className="mb-2",
                    min=10,
                    max=100,
                    step=10,
                    value=100,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Div(className="flex-1"),
                html.Button(
                    "Fit pipeline ✔️",
                    id="fit_pipeline",
                    n_clicks=0,
                    className="""
                        text-sky-700 hover:text-sky-800
                        h-12 w-full flex-0 text-center mt-10
                        bg-gray-400 bg-opacity-5
                        hover:bg-opacity-10
                        transition-all ease-in
                        rounded-xl
                    """,
                ),
            ],
        ),
    ],
)
