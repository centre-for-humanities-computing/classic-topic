"""Module describing the component for inspecting documents"""

from dash import dcc, html

document_inspector = html.Div(
    className="""basis-1/3 flex-0 flex-col bg-white shadow
    overflow-y-scroll overflow-x-hidden p-5
    """,
    children=[
        dcc.Dropdown(
            id="document_selector",
            options={},
            value=None,
        ),
        html.Span(
            className="""
            block mt-2 w-full bg-gray-50 h-0.5
            """
        ),
        html.H2("Information", className="text-lg font-bold p-3"),
        html.Span(
            className="""
            block mb-2 w-full bg-sky-50 h-0.5
            """
        ),
        html.Ul(
            [
                html.Li(
                    id="document_genre",
                    children="TLG genre: None",
                    className="text-lg py-2",
                ),
                html.Li(
                    id="document_group",
                    children="Ground: None",
                    className="text-lg py-2",
                ),
            ],
            className="list-disc pl-8",
        ),
        html.Span(
            className="""
            block mt-2 w-full bg-gray-50 h-0.5
            """
        ),
        html.H2("Topics", className="text-lg font-bold p-3"),
        html.Span(
            className="""
            block mb-2 w-full bg-sky-50 h-0.5
            """
        ),
        dcc.Graph(id="document_topics_graph"),
        html.Span(
            className="""
            block mt-2 w-full bg-gray-50 h-0.5
            """
        ),
        html.H2("Content", className="text-lg font-bold p-3"),
        html.Span(
            className="""
            block mb-2 w-full bg-sky-50 h-0.5
            """
        ),
        html.Div(
            id="document_content",
            children="This is the textual content of the document",
            className="""
            text-justify h-1/3
            """,
        ),
    ],
)
