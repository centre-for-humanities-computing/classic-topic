"""Genre weight popup container component"""
from typing import Dict, List

import dash
import dash_daq as daq
from dash import dcc, html
from dash import ctx
from dash_extensions.enrich import Input, Output, ServersideOutput, State
from dash.exceptions import PreventUpdate

from app.utils.callback import init_callbacks
from app.utils.metadata import fetch_metadata

callbacks, def_callback = init_callbacks()

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


select_deselect = html.Div(
    className="""
        text-lg transition-all ease-in
        justify-center content-center items-center
        flex text-white mx-2 my-5
    """,
    children=[
        html.Button(
            "Select All",
            className="""transition-all ease-in
                text-green-700 hover:text-green-800
                rounded-l-xl bg-green-200
                bg-opacity-10 hover:bg-opacity-20
                pl-3 py-1.5 pr-3
            """,
            id="genre_weights_select_all",
        ),
        html.Button(
            "Deselect All",
            className="""transition-all ease-in
                text-red-700 hover:text-red-800
                rounded-r-xl bg-red-200
                bg-opacity-10 hover:bg-opacity-20
                pr-3 py-1.5 pl-3
            """,
            id="genre_weights_deselect_all",
        ),
    ],
)


layout = html.Div(
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
                        select_deselect,
                        html.Div(className="flex-1"),
                        # html.H2(
                        #     "Genre settings",
                        #     className="text-lg ml-5 flex-1",
                        # ),
                        # html.Div(className="flex-1"),
                        html.Button(
                            "Done",
                            id="close_weight_popup",
                            title="Close popup",
                            className="""
                                text-lg transition-all ease-in
                                justify-center content-center items-center
                                text-sky-700 hover:text-sky-800
                                bg-gray-300 bg-opacity-0
                                hover:bg-opacity-10
                                flex px-3 py-1.5 rounded-xl mx-2
                            """,
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


@def_callback(
    Output("genre_weight_popup", "children"),
    Input("fit_store", "data"),
    prevent_initial_call=True,
)
def update_popup_children(fit_data: Dict) -> List:
    """Updates the children of the genre weights popup"""
    # Fetch metadata where group is present
    md = fetch_metadata().dropna(subset="group")
    # Extract all unique groups + Rest
    genres = md.group.unique().tolist() + ["Rest"]
    # Create an element for each genre
    return [genre_weight_element(genre) for genre in genres]


@def_callback(
    Output(dict(type="genre_switch", index=dash.ALL), "on"),
    State(dict(type="genre_switch", index=dash.ALL), "id"),
    Input("genre_weights_deselect_all", "n_clicks"),
    Input("genre_weights_select_all", "n_clicks"),
)
def toggle_all(
    switch_ids: List[Dict[str, str]], n_deselect: int, n_select: int
) -> List[bool]:
    """Selects or deselects all genres."""
    # If none of the buttons have been clicked so far or there are no genres
    # loaded, prevent from updating
    if (not n_deselect and not n_select) or not switch_ids:
        raise PreventUpdate
    # Getting number of genres
    n_genres = len(switch_ids)
    if ctx.triggered_id == "genre_weights_deselect_all":
        # Return all falses if deselect triggered the callback
        return [False] * n_genres
    if ctx.triggered_id == "genre_weights_select_all":
        # Return all trues if select triggered the callback
        return [True] * n_genres
    # Since we have been exhaustive, this should never happen
    # I'm just pleasing static analysis tools
    raise PreventUpdate


@def_callback(
    Output("genre_weights", "data"),
    Input(dict(type="genre_switch", index=dash.ALL), "on"),
    State(dict(type="genre_switch", index=dash.ALL), "id"),
    Input(dict(type="genre_weight_slider", index=dash.ALL), "value"),
    State(dict(type="genre_weight_slider", index=dash.ALL), "id"),
    prevent_initial_call=True,
)
def update_genre_weights(
    is_on_values: List[bool],
    switch_ids: List[Dict[str, str]],
    weight_values: List[int],
    weight_ids: List[Dict[str, str]],
):
    """Updates genre weights based on the values set in the popup."""
    # If the callback is frivolous, don't update anything
    if not is_on_values or not switch_ids or not weight_values or not weight_ids:
        raise PreventUpdate()
    # Mapping genre indices to their status
    is_on = {id["index"]: on for id, on in zip(switch_ids, is_on_values)}
    # Mapping genre indices to their weights
    weights = {id["index"]: weight for id, weight in zip(weight_ids, weight_values)}
    genres = is_on.keys()
    # For each genre if the genre is ON, map its name to its assigned weight
    # if it isn't, assign a weight of 0
    genre_weights = dict()
    for genre in genres:
        if is_on[genre]:
            genre_weights[genre] = weights[genre]
        else:
            genre_weights[genre] = 0
    return genre_weights


@def_callback(
    Output("genre_weight_popup_container", "className"),
    Input("weight_settings", "n_clicks"),
    Input("close_weight_popup", "n_clicks"),
    prevent_initial_call=True,
)
def open_close_popup(open_clicks: int, close_clicks: int) -> str:
    """Opens and closes genre weights popup when needed"""
    if not open_clicks and not close_clicks:
        raise PreventUpdate()
    if "weight_settings" == ctx.triggered_id:
        return popup_container_class + " fixed"
    if "close_weight_popup" == ctx.triggered_id:
        return popup_container_class + " hidden"
    raise PreventUpdate()


@def_callback(
    Output(dict(type="genre_settings_container", index=dash.MATCH), "className"),
    Input(dict(type="genre_switch", index=dash.MATCH), "on"),
    prevent_initial_call=True,
)
def hide_genre_settings(is_genre_on: bool) -> str:
    """Hides settings for a genre if it is filtered away in the popup"""
    if is_genre_on:
        return setting_container_class
    else:
        return setting_container_class + " hidden"
