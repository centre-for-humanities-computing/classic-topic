import dash
from dash import dcc, html

button_class = """
    text-xl transition-all ease-in 
    justify-center content-center items-center
    text-gray-500 hover:text-sky-600
    flex p-8
"""
popup_container_class = """
    w-full h-full bg-black bg-opacity-10 z-50
"""
genre_weight_popup = html.Div(
    className=popup_container_class + " hidden",
    id="genre_weight_popup_container",
    children=[
        html.Div(
            className="""
                w-1/2 h-1/2 bg-white rounded-xl shadow-md
                absolute flex flex-col
                top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
            """,
            children=[
                html.Div(
                    className="""
                        h-20 w-full rounded-xl bg-gray-50
                        flex flex-row justify-between content-center
                        items-center shadow z-10
                    """,
                    children=[
                        html.H2("Genre weight settings", className="text-xl ml-8"),
                        html.Div(className="flex-1"),
                        html.Button(
                            html.I(className="fa-solid fa-xmark"),
                            id="close_weight_popup",
                            title="Close popup",
                            className=button_class,
                        ),
                    ],
                ),
                html.Div(
                    className="""
                        flex-1 flex-col bg-white rounded-xl
                    """,
                )
            ],
        )
    ],
)
