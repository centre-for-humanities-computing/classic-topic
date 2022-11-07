# type: ignore
"""Component for saving and loading fit data."""

from dash_extensions.enrich import dcc, html

button_class = """
    text-xl transition-all ease-in 
    justify-center content-center items-center
    text-gray-500 hover:text-sky-600
    flex flex-1
"""

save_load = html.Div(
    className="""
        fixed flex flex-none flex-row justify-center content-middle
        left-0.5 bottom-10 h-16 w-32 bg-white shadow rounded-full
        rounded-full ml-5
    """,
    children=[
        html.Button(
            html.I(className="fa-solid fa-file-arrow-down"),
            id="download_button",
            title="Download data",
            className=button_class,
        ),
        dcc.Download(id="download"),
        html.Div(
            className=button_class,
            children=dcc.Upload(
                id="upload",
                children=html.Button(
                    html.I(className="fa-solid fa-file-arrow-up"),
                    id="upload_button",
                    title="Upload data",
                    # className=button_class,
                ),
            ),
        ),
    ],
)
