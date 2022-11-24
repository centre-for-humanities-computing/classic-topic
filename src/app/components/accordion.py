"""Module containing a generic accordion component"""
from typing import Optional, Tuple

import dash
from dash_extensions.enrich import dcc, html
from dash_extensions.enrich import Input, Output, ServersideOutput, State

from app.utils.callback import init_callbacks

callbacks, def_callback = init_callbacks()

# Styles for the accordion body
visible = "flex-1 flex-col flex items-stretch justify-evenly"


# NOTE: consider using Dash Mantine. It's way to much effort to maintain this
# and adds a lot of unnecessary complexity.
def Accordion(
    name: str,
    children,
    index: Optional[str] = None,
    # NOTE: This is 100% implementation detail, Mantine would also
    # elliminate this.
) -> html.Div:
    """
    Dash Accordion component.

    Parameters
    ----------
    name: str
        Name that will be displayed on the top of the accordion.
    children: list of dash component
        List of children of the accordion.
    index: str or None, default None
        Index to be used in the callbacks.
        If not specified, name will be used as index.

    Note
    ----
    In order for the component to work properly, you should add the
    accordion's callbacks to the global callback list.
    """
    if index is None:
        index = name
    return html.Div(
        children=[
            html.Div(
                [
                    html.H3(
                        name,
                        className="text-xl",
                    ),
                    html.Span(className="flex-1"),
                    html.Button(
                        html.I(
                            className="fa-solid fa-chevron-up",
                        ),
                        id=dict(type="_setting_group_collapse", index=index),
                        n_clicks=0,
                    ),
                ],
                className="""
                flex flex-row justify-center content-center
                px-5 py-1
                """,
            ),
            html.Span(
                className="""
                block bg-gray-500 bg-opacity-10 h-0.5 self-center m-2
                """
            ),
            html.Div(
                id=dict(type="_setting_group_body", index=index),
                className=visible,
                children=children,
            ),
        ],
        className="""
            justify-center content-center
            bg-white p-3 rounded-2xl shadow
            transition-all ease-in 
        """,
    )


def AccordionItem(name: str, *children):
    return html.Div(
        className="""
            flex flex-row flex-1
            justify-between justify-items-stretch
            content-center items-center
            px-4 my-1.5
            """,
        children=[
            html.P(name),
            *children,
        ],
    )


@def_callback(
    Output(
        dict(type="_setting_group_body", index=dash.MATCH),
        "className",
    ),
    Output(
        dict(type="_setting_group_collapse", index=dash.MATCH),
        "className",
    ),
    Input(dict(type="_setting_group_collapse", index=dash.MATCH), "n_clicks"),
    prevent_initial_call=True,
)
def toggle_accordion(n_clicks: int) -> Tuple[str, str]:
    """Toggles accordion"""
    # If the number of times the expander has been clicked is not divisible
    # by two, we know the accordion body is visible
    is_on = not (n_clicks % 2)
    if is_on:
        # Turning the accordion body visible
        # Rotating the collapser
        return visible, "transition-all ease-in rotate-0"
    else:
        # Hiding accordion body
        # Rotating the collapser
        return "hidden", "transition-all ease-in rotate-180"
