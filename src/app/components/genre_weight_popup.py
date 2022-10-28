import dash
import dash_daq as daq
from dash import dcc, html

button_class = """
    text-lg transition-all ease-in 
    justify-center content-center items-center
    text-sky-700 hover:text-sky-800
    bg-gray-300 bg-opacity-0
    hover:bg-opacity-10
    flex px-3 py-1.5 rounded-xl mx-2 my-3
"""
popup_container_class = """
    w-full h-full bg-black bg-opacity-10 z-50
"""
setting_container_class = """
    flex flex-row justify-around content-center items-stretch
    my-4 p-5 rounded-b-xl
"""


def genre_weight_element(genre_name: str) -> html.Div:
    """Creates an element for a given genre, with which it can be filtered or
    its weight can be changed.

    Parameters
    ----------
    genre_name: str
        Name of the genre you want to create an element for.

    Returns
    -------
    Div
        Dash HTML component.
    """
    return html.Div(
        children=[
            html.Div(
                [
                    html.H3(
                        id=dict(type="genre_name", index=genre_name),
                        children=genre_name,
                        className="text-xl",
                    ),
                    html.Span(className="flex-1"),
                    daq.BooleanSwitch(
                        id=dict(
                            type="genre_switch",
                            index=genre_name,
                        ),
                        on=True,
                        color="#16a34a",
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
                className=setting_container_class,
                children=[
                    html.P("Weight: "),
                    dcc.Slider(
                        1,
                        1000,
                        step=None,
                        id=dict(
                            type="genre_weight_slider",
                            index=genre_name,
                        ),
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                        },
                        marks={
                            number: str(number)
                            for number in [1, 10, 50, 100, 200, 500, 1000]
                        },
                        value=1,
                        className="""
                        flex-1
                        """,
                    ),
                ],
                id=dict(type="genre_settings_container", index=genre_name),
            ),
        ],
        className="""
            justify-center content-center
            bg-white p-3 rounded-2xl shadow
            my-3 transition-all ease-in 
        """,
    )


genre_weight_popup = html.Div(
    className=popup_container_class + " hidden",
    id="genre_weight_popup_container",
    children=[
        html.Div(
            className="""
                w-1/3 h-2/3 bg-white rounded-xl shadow-md
                absolute flex flex-col
                top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
            """,
            children=[
                html.Div(
                    className="""
                        h-14 w-full rounded-t-xl 
                        bg-white
                        flex flex-row justify-between content-center
                        items-center shadow z-10
                    """,
                    children=[
                        html.H2(
                            "Genre settings",
                            className="text-lg ml-5 flex-1",
                        ),
                        html.Div(className="flex-1"),
                        html.Button(
                            "Done",
                            id="close_weight_popup",
                            title="Close popup",
                            className=button_class,
                        ),
                    ],
                ),
                html.Div(
                    className="""
                        flex-1 rounded-xl
                        bg-gray-50
                        overflow-y-scroll px-5
                        flex-col
                    """,
                    id="genre_weight_popup",
                ),
            ],
        )
    ],
)
