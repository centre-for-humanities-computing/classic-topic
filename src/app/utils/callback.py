from typing import Dict, List, Tuple, Callable

import dash


def add_callbacks(app: dash.Dash, callbacks: List[Dict]) -> None:
    """Adds the list of callbacks to a Dash app.

    Parameters
    ----------
    app: Dash
        Dash application to add callbacks to.
    callbacks: list of dict
        Callback list to add to the app.
    """
    for callback in callbacks:
        app.callback(*callback["args"], **callback["kwargs"])(callback["function"])


def init_callbacks() -> Tuple[List[Dict], Callable]:
    """Initialises callbacks for a module.

    Returns
    -------
    callbacks: list of dict
        List of callbacks for the module, that can be added to an app.
    decorator: function
        Function decorator that will add the function to the callback list as
        a callback.
    """
    callbacks = []

    def decorator(*args, **kwargs) -> Callable:
        def _cb(func: Callable):
            callbacks.append({"function": func, "args": args, "kwargs": kwargs})
            return func

        return _cb

    return callbacks, decorator
