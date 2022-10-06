invisible = {"display": "none"}

transparent = {"opacity": "0"}
opaque = {"opacity": "1"}

sidebar = {
    # "padding": "10px",
    "display": "flex",
    "flex-direction": "row",
    "margin": "0",
    "flex": "1 0",
    "background": "#ffffff",  # "#fcfcfc",
    # "box-shadow": "-2px 0 5px #00000066",
    "z-index": "5",
    "right": "500px",
}

sidebar_collapsed = {**sidebar, "flex": "0 0 40px"}

sidebar_body = {
    "padding": "15px",
    "display": "flex",
    "flex-direction": "column",
    "margin": "0",
    "flex": "8 0",
    "background": "#ffffff",
    # "box-shadow": "-2px 0 5px #00000066",
    "z-index": "5",
}

collapse_button = {
    "display": "flex",
    "flex": "1 0",
    "background": "#fcfcfc",
    "padding": "10px",
    "outline": "none",
    "border": "none",
    "justify-content": "center",
    "align-items": "center",
    "font": "bold 20px Helvetica",
    "color": "#bbbbbb",
}

page_visible = {
    # 'padding': '10px',
    "display": "block",
    # "flex-direction": "column",
    "flex": "4",
    "height": "100%",
    "font": "15px Helvetica",
    "overflow-y": "auto",
    "padding": "10px",
}


window = {
    "top": "0",
    "left": "0",
    "display": "flex",
    "flex-direction": "row",
    "justify-content": "space-around",
    "align-items": "stretch",
    # "flex": "1",
    "height": "100%",
    "width": "100%",
    "position": "fixed",
}

submit = {
    "margin-top": "10px",
    "padding": "15px",
    "padding-left": "20px",
    "padding-right": "20px",
    "outline": "false",
    "font": "bold 15px Helvetica",
    "background": "#8100d1",
    "color": "white",
    "border": "none",
    "border-radius": "30px",
}

topic_switcher = {
    "flex": "1 0",
    "background": "white",
    # "border": "solid 3px #777777",
    "border-radius": "30px",
    "margin": "20px",
    # "margin-bottom": "30px",
    "z-index": "8",
    "display": "flex",
    "flex-direction": "row",
    "justify-content": "space-around",
    "align-items": "stretch",
    "border-bottom": "solid 2px #dddddd",
}

topic_namer = {
    "font": "18px Helvetica",
    "text-align": "center",
    "border": "none",
    "color": "#000000",
    "background-color": "#00000000",
    "margin-left": "10px",
    "border-bottom": "solid 2px #8100d1",
}

next_prev_topic = {
    "font": "18px Helvetica",
    "background-color": "#00000000",
    "color": "#555555",
    "border": "none",
    "flex": "3 1",
}
