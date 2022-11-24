from typing import Tuple, Dict

from dash import ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import dcc, html
from dash_extensions.enrich import Input, Output, State

from app.components import accordion
from app.components.toolbar import toolbar_class
from app.utils.callback import init_callbacks

callbacks, def_callback = init_callbacks()

sidebar_body_class = """
    flex fixed w-1/3 flex-col p-3 h-full
    transition-all ease-in right-0 pr-5 z-30
    space-y-3
    bg-black bg-opacity-0 
"""

sidebar_shade_class = """
    fixed w-full h-full bg-black 
    top-0 left-0
    z-10 transition-all ease-in
"""


feature_extraction = accordion.Accordion(
    name="Feature extraction",
    children=[
        accordion.AccordionItem(
            "Vectorizer: ",
            html.Div(
                dcc.Dropdown(
                    id="select_vectorizer",
                    className="flex-1",
                    options={
                        "bow": "Count",
                        "tf-idf": "TF-IDF",
                    },
                    value="bow",
                ),
                className="flex-1 ml-5",
            ),
        ),
        accordion.AccordionItem(
            "N-gram range: ",
            dcc.RangeSlider(
                1,
                7,
                step=1,
                value=[1, 1],
                allowCross=False,
                id="n_gram_slider",
                className="flex-1 mt-5",
            ),
        ),
        accordion.AccordionItem(
            "Max doc frequency: ",
            dcc.Slider(
                id="max_df",
                min=0.0,
                max=1.0,
                step=0.1,
                value=1.0,
                tooltip={"placement": "bottom", "always_visible": True},
                className="flex-1 mt-5",
            ),
        ),
        accordion.AccordionItem(
            "Min doc occurrance: ",
            dcc.Slider(
                id="min_df",
                min=0,
                max=50,
                step=1,
                value=0,
                marks={i * 10: str(i * 10) for i in range(6)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="flex-1 mt-5",
            ),
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
    ],
)

topic_model = accordion.Accordion(
    name="Topic model",
    children=[
        accordion.AccordionItem(
            "Vectorizer: ",
            html.Div(
                dcc.Dropdown(
                    id="select_model",
                    className="flex-1",
                    options={
                        "nmf": "Non-negative Matrix Factorization",
                        "lda": "Latent Dirichlet Allocation",
                        "lsa": "Latent Semantic Allocation/Indexing",
                        "dmm": "Dirichlet Multinomial Mixture",
                    },
                    value="nmf",
                ),
                className="flex-1 ml-5",
            ),
        ),
        accordion.AccordionItem(
            "Number of topics: ",
            dcc.Slider(
                id="n_topics",
                className="flex-1 mt-5",
                min=10,
                max=100,
                step=10,
                value=100,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ),
    ],
)

layout = html.Div(
    id="sidebar",
    children=[
        html.Div(
            className="""
            fixed flex-0 flex-row justify-end content-around
            top-0 right-0 z-0 m-3 mr-4
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
            id="sidebar_shade",
            className="hidden",
        ),
        html.Div(
            id="sidebar_body",
            className=sidebar_body_class + " translate-x-full",
            children=[
                feature_extraction,
                topic_model,
                html.Div(
                    className="""
                    flex flex-row space-x-2
                    """,
                    children=[
                        html.Button(
                            children=[
                                "Cancel ",
                                html.Span(className="w-3"),
                                html.I(className="fa-solid fa-ban"),
                            ],
                            id="cancel_pipeline",
                            n_clicks=0,
                            className="""
                                basis-1/3
                                text-black bg-red-400
                                hover:bg-red-500
                                h-12 flex-1 text-center 
                                transition-all ease-in
                                rounded-xl shadow
                            """,
                        ),
                        html.Button(
                            children=[
                                "Fit pipeline ",
                                html.Span(className="w-3"),
                                html.I(className="fa-solid fa-check"),
                            ],
                            id="fit_pipeline",
                            n_clicks=0,
                            className="""
                                basis-2/3
                                text-white bg-green-600
                                hover:bg-green-700
                                h-12 flex-1 text-center 
                                transition-all ease-in
                                rounded-xl shadow
                            """,
                        ),
                    ],
                ),
            ],
        ),
    ],
)
sidebar = layout


@def_callback(
    Output("sidebar_body", "className"),
    Output("topic_toolbar", "className"),
    Output("sidebar_shade", "className"),
    Input("sidebar_collapser", "n_clicks"),
    Input("current_view", "data"),
    prevent_initial_call=True,
)
def open_close_sidebar(n_clicks: int, current_view: str) -> Tuple[str, str, str]:
    """Opens or closes sidebar and moves and hides the topic switcher."""
    hide_switcher = " translate-y-0" if current_view == "topic" else " translate-y-full"
    is_open = (n_clicks % 2) == 0

    if is_open:
        return (
            sidebar_body_class + " translate-x-full",
            toolbar_class + hide_switcher,
            sidebar_shade_class + " bg-opacity-0 hidden",
        )
    else:
        return (
            sidebar_body_class + " translate-x-0",
            toolbar_class + hide_switcher,
            sidebar_shade_class + " bg-opacity-40 block",
        )


@def_callback(
    Output("sidebar_collapser", "n_clicks"),
    State("sidebar_collapser", "n_clicks"),
    Input("fit_store", "data"),
    Input("fit_pipeline", "n_clicks"),
    Input("cancel_pipeline", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_sidebar_fitting(
    current: int, fit_data: Dict, fit_clicks: int, cancel_clicks: int
) -> int:
    """Opens or closes sidebar when the fit pipeline button is pressed."""
    # TODO: Explain what this does, this is not understandable without
    # documentation
    if (
        ((ctx.triggered_id == "fit_store") and (fit_data is None))
        or ((ctx.triggered_id == "fit_pipeline") and fit_clicks)
        or ((ctx.triggered_id == "cancel_pipeline") and cancel_clicks)
    ):
        return current + 1
    else:
        raise PreventUpdate()
