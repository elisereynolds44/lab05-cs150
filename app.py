from dash import Dash
from layout.main_layout import layout
from callbacks.svm_callbacks import register_callbacks

from utils.data import generate_arsenal_data
import utils.figures as figs

app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Arsenal Match Predictor"
app.layout = layout
server = app.server

register_callbacks(app, figs, generate_arsenal_data)

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
