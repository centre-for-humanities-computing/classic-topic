import dash
from app.callbacks import add_callbacks
from app.layout import add_layout

app = dash.Dash(__name__, title="Topic Modelling")

add_layout(app)
add_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True, port=8080, host="0.0.0.0")
