"""Module describing the component for inspecting documents"""

from dash_extensions.enrich import dcc, html

from app.components.settings import setting_group

document_inspector = html.Div(
    className="""basis-1/3 flex-1 flex-col bg-white shadow
    overflow-y-scroll overflow-x-hidden p-5 space-y-5
    """,
    children=[
        dcc.Dropdown(
            id="document_selector",
            options={},
            value=None,
        ),
        setting_group(
            name="Information",
            index="inspector_info",
            children=[
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
            ],
        ),
        setting_group(
            "Topics",
            index="inspector_topics",
            children=[
                dcc.Graph(id="document_topics_graph", animate=False),
            ],
        ),
        setting_group(
            "Content",
            index="inspector_content",
            children=[
                html.Div(
                    id="document_content",
                    children="This is the textual content of the document",
                    className="""
                text-justify h-1/3
                """,
                ),
            ],
        ),
    ],
)
