from typing import List, Optional

from dash import dcc, html

settings_visible = "flex-1 flex-col flex items-stretch justify-evenly"
settings_hidden = "hidden"


def setting_group(
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
                className=settings_visible,
                children=children,
            ),
        ],
        className="""
            justify-center content-center
            bg-white p-3 rounded-2xl shadow
            transition-all ease-in 
        """,
    )


def setting(name: str, *children):
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
