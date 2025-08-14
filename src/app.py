import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_templates import load_figure_template


# Minimal Dash app configured for Dash Pages
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.join(os.path.dirname(__file__), 'pages'),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
load_figure_template("cyborg")


def _nav_links() -> dbc.Nav:
    return dbc.Nav(
        [
            dbc.NavLink("Draft", href="/", active="exact"),
            dbc.NavLink("Regular Season", href="/regular", active="exact"),
            dbc.NavLink("Post Season", href="/post", active="exact"),
        ],
        pills=True,
        className="mb-4",
    )


app.layout = dbc.Container([
    _nav_links(),
    dash.page_container,
], fluid=True, className="py-4 px-5")

if __name__ == '__main__':
    app.run_server(debug=True) 