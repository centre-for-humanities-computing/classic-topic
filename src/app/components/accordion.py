from typing import Optional, Tuple

import dash
from dash_extensions.enrich import dcc, html
from dash_extensions.enrich import Input, Output, ServersideOutput, State

from app.utils.callback import init_callbacks

callbacks, def_callback = init_callbacks()

visible = "flex-1 flex-col flex items-stretch justify-evenly"
hidden = "hidden"


def Accordion(
    name: str,
    children,
    index: Optional[str] = None,
) -> html.Div:
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
def expand_hide_setting_group(n_clicks: int) -> Tuple[str, str]:
    is_on = not (n_clicks % 2)
    if is_on:
        return visible, "transition-all ease-in rotate-0"
    else:
        return "hidden", "transition-all ease-in rotate-180"
