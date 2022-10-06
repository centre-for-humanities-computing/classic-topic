import dash

app = dash.Dash(
    __name__,
    title="Topic Modeling",
    external_scripts={"src": "https://cdn.tailwindcss.com"},
)
