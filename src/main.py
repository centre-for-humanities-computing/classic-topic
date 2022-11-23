"""Main script creating the app, adding layout and callbacks and running the server"""
from dash_extensions.enrich import Dash

from app.app import layout, callbacks
from app.utils.callback import add_callbacks

app = Dash(
    __name__,
    title="Topic Modelling",
    external_scripts=[
        {
            "src": "https://cdn.tailwindcss.com",
        },
        {
            "src": "https://kit.fontawesome.com/9640e5cd85.js",
            "crossorigin": "anonymous",
        },
    ],
    url_base_pathname="/classic-topic/",
)

app.layout = layout
for callback in callbacks:
    print(callback["function"])
add_callbacks(app, callbacks)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True, port=8080, host="0.0.0.0")
