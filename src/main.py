"""Main script creating the app, adding layout and callbacks and running the server"""
from dash_extensions.enrich import Dash

from app.callbacks import add_callbacks
from app.layout import add_layout

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
)

add_layout(app)
add_callbacks(app)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True, port=8080, host="0.0.0.0")
