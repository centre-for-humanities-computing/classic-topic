# type: ignore
from dash import dcc, html

sidebar_body_class = """
    flex fixed w-3/12 h-full flex-col p-8 bg-white shadow-lg rounded-xl
    transition-all ease-in space-y-3 top-0 right-0 z-30
"""

sidebar = html.Div(
    id="sidebar",
    children=[
        html.Div(
            className="""
            fixed flex-0 flex-row justify-end content-around
            top-0 right-0 z-40 m-3 mr-4
            """,
            children=html.Button(
                html.I(className="fa-solid fa-gears"),
                id="sidebar_collapser",
                n_clicks=0,
                className="""
                    flex justify-center content-center
                    flex-0 h-12 m-1 mt-5
                    text-2xl text-center text-gray-500
                    hover:text-sky-700
                """,
            ),
        ),
        html.Div(
            id="sidebar_body",
            className=sidebar_body_class + " translate-x-full",
            children=[
                #  html.H1(
                #      "Pipeline Settings",
                #      className="text-2xl mt-2 mb-3",
                #  ),
                html.Span(
                    className="""
                    block w-full bg-gray-50 h-0.5
                    """
                ),
                # html.Span(className="h-5"),
                html.H2("Feature extraction", className="text-lg font-bold"),
                html.Span(
                    className="""
                    block w-full bg-sky-50 h-0.5
                    """
                ),
                html.H3("Vectorizer", className="italic text-base "),
                dcc.Dropdown(
                    id="select_vectorizer",
                    className="mx-2",
                    options={
                        "bow": "Count",
                        "tf-idf": "TF-IDF",
                    },
                    value="bow",
                ),
                html.H3(
                    "N-gram range",
                    className="italic text-base ",
                ),
                dcc.RangeSlider(
                    1,
                    7,
                    step=1,
                    value=[1, 1],
                    allowCross=False,
                    id="n_gram_slider",
                ),
                html.H3(
                    "Maximum feature document frequency",
                    className="italic text-base ",
                ),
                dcc.Slider(
                    id="max_df",
                    className="",
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    value=0.1,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H3(
                    "Minimum feature document occurrance",
                    className="italic text-base ",
                ),
                dcc.Slider(
                    id="min_df",
                    className="",
                    min=0,
                    max=50,
                    step=1,
                    value=10,
                    marks={i * 10: str(i * 10) for i in range(6)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Button(
                    html.H3(
                        "Genre filtering/weights...",
                        className="italic text-base ",
                    ),
                    n_clicks=0,
                    id="weight_settings",
                    className="""
                        text-sky-700
                        hover:text-sky-800
                        p-1
                    """,
                ),
                html.Span(
                    className="""
                    block w-full bg-gray-50 h-0.5
                    """
                ),
                html.H2("Topic model", className="text-lg font-bold"),
                html.Span(
                    className="""
                    block w-full bg-sky-50 h-0.5
                    """
                ),
                dcc.Dropdown(
                    id="select_model",
                    className="mx-2",
                    options={
                        "nmf": "Non-negative Matrix Factorization",
                        "lda": "Latent Dirichlet Allocation",
                        "lsa": "Latent Semantic Allocation/Indexing",
                        "dmm": "Dirichlet Multinomial Mixture",
                    },
                    value="nmf",
                ),
                html.H3("Number of topics", className="italic text-base "),
                dcc.Slider(
                    id="n_topics",
                    className="",
                    min=10,
                    max=100,
                    step=10,
                    value=100,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Div(className="flex-1"),
                html.Button(
                    children=[
                        "Fit pipeline ",
                        html.Span(className="w-3"),
                        html.I(className="fa-solid fa-check"),
                    ],
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
