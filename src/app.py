import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_templates import load_figure_template
import logging
from typing import Optional


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
    # Configure logging and run startup refresh of data feeds
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _log_df_shape(name: str, df) -> None:
        try:
            shape = getattr(df, 'shape', None)
            logging.info("%s shape: %s", name, shape)
        except Exception:
            pass

    def _startup_refresh() -> None:
        from src.data_loader import get_google_sheets_data, get_regular_season_data
        logging.info("Starting data feed refresh on startup...")
        # Draft-related sheets
        try:
            draft_data = get_google_sheets_data()
            logging.info("Draft sheets connected successfully (%d tabs)", len(draft_data))
        except Exception as exc:
            logging.error("Draft sheets connection failed: %s", exc)
        # Regular season sheets
        try:
            rs_data = get_regular_season_data()
            logging.info("Regular season sheets connected successfully")
            for key in ('standings', 'team_stats', 'player_stats'):
                if key in rs_data:
                    _log_df_shape(f"regular.{key}", rs_data[key])
        except Exception as exc:
            logging.error("Regular season sheets connection failed: %s", exc)

    _startup_refresh()
    app.run_server(debug=True) 