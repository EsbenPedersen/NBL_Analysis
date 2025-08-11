import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, no_update, ctx
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Any, Optional, List, Dict, Tuple
from io import StringIO

from src.data_loader import get_google_sheets_data, get_regular_season_data
from src.data_processing import process_data
from src.draft_optimizer import get_draft_recommendations

# Initialize the Dash app with viewport meta tag for responsive design
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.CYBORG], 
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server
load_figure_template("cyborg")

# --- Helper Functions ---
def generate_heatmap_style(df: DataFrame, column: str, n_bins: int = 7, colors: list[str] | None = None) -> list[dict]:
    """
    Generates conditional styling for a table column to create a heatmap effect.
    """
    if colors is None:
        # A green scale from light to dark
        colors = ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
    
    styles = []
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return styles
        
    col_data = df[column].dropna()
    if col_data.empty:
        return styles
        
    min_val = col_data.min()
    max_val = col_data.max()
    
    if min_val == max_val:
        styles.append({
            'if': {'column_id': column},
            'backgroundColor': colors[len(colors) // 2],
            'color': 'white'
        })
        return styles
        
    bounds = np.linspace(min_val, max_val, n_bins + 1)
    
    for i in range(n_bins):
        min_bound = bounds[i]
        max_bound = bounds[i+1]
        
        if i == n_bins - 1:
            filter_q = f'{{{column}}} >= {min_bound} && {{{column}}} <= {max_bound}'
        else:
            filter_q = f'{{{column}}} >= {min_bound} && {{{column}}} < {max_bound}'
            
        bg_color = colors[i]
        text_color = 'white' if i >= n_bins / 2 else 'black'
        
        styles.append({
            'if': {
                'filter_query': filter_q,
                'column_id': column
            },
            'backgroundColor': bg_color,
            'color': text_color
        })
    return styles

def create_slider_marks(min_val, max_val, num_marks=10):
    """Create evenly spaced, rounded marks for a slider."""
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return {0: '0'}
    step_values = np.linspace(min_val, max_val, num_marks)
    return {int(value): '{:.2f}'.format(value) for value in step_values}

def preprocess_size_data(size_data):
    if size_data.empty or size_data.isnull().all():
        return pd.Series(10, index=size_data.index)
    
    # Fill NaNs with the mean to avoid issues in calculation
    size_data = size_data.fillna(size_data.mean())
    if size_data.isnull().all(): # If all were NaN, mean is NaN, so check again
        return pd.Series(10, index=size_data.index)

    min_val = size_data.min()
    if min_val < 0:
        size_data = size_data + abs(min_val)
    
    min_val = size_data.min() # Recalculate min_val after potential shift
    max_val = size_data.max()

    if pd.isna(max_val) or pd.isna(min_val) or max_val == min_val:
        return pd.Series(10, index=size_data.index)

    size_data = 10 + ((size_data - min_val) / (max_val - min_val)) * 40
    return size_data.fillna(10) # Fill any remaining NaNs just in case

def build_draft_layout() -> Any:
    return html.Div([
        dcc.Store(id='data-store'),
        dcc.Store(id='filters-store', data={}),
        dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0),

        # Header
        dbc.Row([
            dbc.Col(html.H1("NBL Draft Analysis"), width='auto'),
            dbc.Col(dbc.Button("Refresh Data", id="refresh-data-btn", color="secondary"), width='auto', className="align-self-center"),
            dbc.Col(dbc.Button("Reset Filters", id="reset-filters-btn", color="light"), width='auto', className="align-self-center")
        ], align="center", className="mb-4"),

        # Main Controls and Scatter Plot
        dbc.Card(
            dbc.CardBody([
                html.Div(id='controls-container'),
                dcc.Graph(id='scatter-plot', style={'height': '60vh'}),
            ]),
            className="mb-4",
        ),

        # Draft Strategy
        dbc.Card(
            [
                dbc.CardHeader(html.H2("Draft Strategy")),
                dbc.CardBody([
                    dbc.Button("Generate Draft Strategy", id="generate-strategy-btn", color="primary", className="mb-3"),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H3("Ideal Team (Top 2 per Position)"),
                            html.Div(id='ideal-team-container')
                        ]), md=6),
                        dbc.Col(html.Div([
                            html.H3("Prioritized Draft Board (by Scarcity)"),
                            html.Div(id='draft-board-container')
                        ]), md=6),
                    ]),
                ]),
            ],
            className="mb-4",
        ),

        # Player Filters (Collapsible)
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(
                        html.H2(
                            dbc.Button(
                                "Player Filters",
                                id="toggle-filters-btn",
                                color="link",
                                n_clicks=0,
                                className="text-decoration-none"
                            ),
                            className="mb-0"
                        )
                    ),
                    dbc.Col(
                        dbc.Button("Reset Player Filters", id="reset-sliders-btn", color="light", size="sm"),
                        width="auto",
                        className="align-self-center"
                    )
                ], justify="between", align="center")
            ),
            dbc.Collapse(
                dbc.CardBody(html.Div(id='sliders-container')),
                id="filters-collapse",
                is_open=False,
            ),
        ]),
    ])


def build_regular_season_layout() -> Any:
    return html.Div([
        dcc.Store(id='rs-data-store'),
        dcc.Interval(id='rs-interval', interval=60 * 1000, n_intervals=0),
        dbc.Row([
            dbc.Col(html.H1("Regular Season"), width='auto'),
            dbc.Col(dbc.Button("Refresh Season Data", id="rs-refresh-btn", color="secondary"), width='auto', className="align-self-center"),
        ], align="center", className="mb-4"),

        # Overview: Standings and Power Rankings
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H3("Standings")),
                dbc.CardBody(html.Div(id='rs-standings-table'))
            ]), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H3("Power Rankings")),
                dbc.CardBody(html.Div(id='rs-power-table'))
            ]), md=6),
        ], className="mb-4"),

        # Top Players
        dbc.Card([
            dbc.CardHeader(html.H3("Top Players")),
            dbc.CardBody(html.Div(id='rs-top-players-table'))
        ], className="mb-4"),

        # Interactive Plots: Teams and Players
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Team Statistics Plot")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='rs-team-x', placeholder='X Axis'), md=3),
                        dbc.Col(dcc.Dropdown(id='rs-team-y', placeholder='Y Axis'), md=3),
                        dbc.Col(dcc.Dropdown(id='rs-team-color', placeholder='Color By', clearable=True), md=3),
                        dbc.Col(dcc.Dropdown(id='rs-team-size', placeholder='Size By', clearable=True), md=3),
                    ], className='mb-2 g-2'),
                    dcc.Graph(id='rs-team-graph', style={'height': '50vh'})
                ])
            ]), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Player Statistics Plot")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='rs-player-x', placeholder='X Axis'), md=3),
                        dbc.Col(dcc.Dropdown(id='rs-player-y', placeholder='Y Axis'), md=3),
                        dbc.Col(dcc.Dropdown(id='rs-player-color', placeholder='Color By', clearable=True), md=3),
                        dbc.Col(dcc.Dropdown(id='rs-player-size', placeholder='Size By', clearable=True), md=3),
                    ], className='mb-2 g-2'),
                    dcc.Graph(id='rs-player-graph', style={'height': '50vh'})
                ])
            ]), md=6),
        ], className="mb-4"),

        # Trade Suggestions
        dbc.Card([
            dbc.CardHeader(html.H3("Trade Suggestions (Simple)")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='rs-team-select', placeholder='Select your team')), 
                    dbc.Col(dbc.Button("Generate Suggestions", id="rs-trade-btn", color="primary"), width='auto')
                ], className='mb-3 g-2'),
                html.Div(id='rs-trade-suggestions')
            ])
        ], className="mb-4"),
    ])


def build_post_season_layout() -> Any:
    return html.Div([
        html.H1("Post Season"),
        html.P("Coming soon...")
    ])


# --- App Layout ---
app.layout = dbc.Container([
    dcc.Tabs(id='tabs', value='draft', children=[
        dcc.Tab(label='Draft', value='draft'),
        dcc.Tab(label='Regular Season', value='regular'),
        dcc.Tab(label='Post Season', value='post'),
    ]),
    html.Div(id='page-content')
], fluid=True, className="py-4 px-5")

# --- Callbacks ---
@app.callback(
    Output('page-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab: str) -> Any:
    if tab == 'draft':
        return build_draft_layout()
    if tab == 'regular':
        return build_regular_season_layout()
    return build_post_season_layout()

@app.callback(
    Output("filters-collapse", "is_open"),
    [Input("toggle-filters-btn", "n_clicks")],
    [State("filters-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_filters_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('data-store', 'data'),
    [Input('interval-component', 'n_intervals'),
     Input('refresh-data-btn', 'n_clicks')]
)
def update_data_store(n_intervals, n_clicks):
    """Fetches and processes data at regular intervals or on button click."""
    raw_data = get_google_sheets_data()
    processed_data = process_data(raw_data)
    # Fill NaNs for JSON compatibility and calculations
    for col in processed_data.columns:
        if pd.api.types.is_numeric_dtype(processed_data[col]):
            processed_data[col] = processed_data[col].fillna(0)
    return processed_data.to_json(date_format='iso', orient='split')

@app.callback(
    Output('filters-store', 'data'),
    [Input('x-column', 'value'), Input('y-column', 'value'), Input('color-column', 'value'),
     Input('size-column', 'value'), Input('name-dropdown', 'value'), Input('position-dropdown', 'value'),
     Input('team-dropdown', 'value'), Input('toggle-text-labels', 'value'), Input('include-drafted-toggle', 'value'),
     Input('exclude-undrafted-toggle', 'value'), Input('exclude-zeros-toggle', 'value'),
     Input('average-by-color-toggle', 'value'),
     Input({'type': 'filter-slider', 'index': dash.ALL}, 'value'),
     Input({'type': 'min-input', 'index': dash.ALL}, 'value'),
     Input({'type': 'max-input', 'index': dash.ALL}, 'value')],
    [State({'type': 'filter-slider', 'index': dash.ALL}, 'id'),
     State({'type': 'min-input', 'index': dash.ALL}, 'id'),
     State({'type': 'max-input', 'index': dash.ALL}, 'id'),
     State('filters-store', 'data')],
    prevent_initial_call=True
)
def update_filters_store(x, y, color, size, names, positions, teams, text, drafted, exclude_undrafted, exclude_zeros, avg_by_color,
                         slider_values, min_inputs, max_inputs,
                         slider_ids, min_input_ids, max_input_ids,
                         current_filters):
    current_filters = current_filters or {}
    triggered_id = ctx.triggered_id
    
    slider_state = current_filters.get('sliders', {})
    
    # Create maps for easy lookup
    slider_map = {s_id['index']: s_val for s_id, s_val in zip(slider_ids, slider_values)}
    
    # Helper to convert input values to float, handling None or empty strings
    def to_float(val):
        if val is None or val == '':
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    min_map = {m_id['index']: to_float(m_val) for m_id, m_val in zip(min_input_ids, min_inputs)}
    max_map = {m_id['index']: to_float(m_val) for m_id, m_val in zip(max_input_ids, max_inputs)}

    if isinstance(triggered_id, dict):
        tr_type = triggered_id.get('type')
        tr_index = triggered_id.get('index')
        
        if tr_type == 'filter-slider':
            slider_state[tr_index] = slider_map.get(tr_index)
        
        elif tr_type == 'min-input':
            new_min = min_map.get(tr_index)
            if new_min is not None:
                current_range = slider_state.get(tr_index, slider_map.get(tr_index, [None, None]))
                current_max = current_range[1] if isinstance(current_range, list) and len(current_range) > 1 else None
                if current_max is not None:
                    if new_min > current_max:
                        new_min = current_max
                    slider_state[tr_index] = [new_min, current_max]
        
        elif tr_type == 'max-input':
            new_max = max_map.get(tr_index)
            if new_max is not None:
                current_range = slider_state.get(tr_index, slider_map.get(tr_index, [None, None]))
                current_min = current_range[0] if isinstance(current_range, list) and len(current_range) > 0 else None
                if current_min is not None:
                    if new_max < current_min:
                        new_max = current_min
                    slider_state[tr_index] = [current_min, new_max]
    else:
        current_filters.update({
            'x_val': x, 'y_val': y, 'color_val': color, 'size_val': size,
            'name_val': names, 'pos_val': positions, 'team_val': teams,
            'text_val': text, 'drafted_val': drafted, 'exclude_undrafted_val': exclude_undrafted, 'exclude_zeros_val': exclude_zeros,
            'avg_by_color_val': avg_by_color
        })
    
    current_filters['sliders'] = slider_state
    
    return current_filters

@app.callback(
    Output('filters-store', 'data', allow_duplicate=True),
    Input('reset-filters-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters_store(n_clicks):
    return {}

@app.callback(
    Output('filters-store', 'data', allow_duplicate=True),
    [Input('reset-sliders-btn', 'n_clicks')],
    [State('filters-store', 'data')],
    prevent_initial_call=True
)
def reset_sliders(n_clicks, current_filters):
    if not n_clicks:
        return no_update
    
    current_filters = current_filters or {}
    if 'sliders' in current_filters:
        current_filters['sliders'] = {}
        
    return current_filters

@app.callback(
    Output('controls-container', 'children'),
    [Input('data-store', 'data'), Input('filters-store', 'data')]
)
def update_controls(json_data, filters):
    if not json_data: return []
    filters = filters or {}
    
    defaults = {'x_val': 'VORP', 'y_val': 'Game Score', 'color_val': 'A/TO', 'size_val': 'PPM',
                'name_val': [], 'pos_val': [], 'team_val': [], 'text_val': [], 'drafted_val': [], 'exclude_undrafted_val': [], 'exclude_zeros_val': ['exclude'],
                'avg_by_color_val': []}
    
    df = pd.read_json(StringIO(json_data), orient='split')
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    all_cols_options = [{'label': col, 'value': col} for col in df.columns]
    name_options = [{'label': 'All', 'value': 'all'}] + [{'label': name, 'value': name} for name in sorted(df['Name'].unique())]
    position_options = [{'label': 'All', 'value': 'all'}] + [{'label': pos, 'value': pos} for pos in sorted(df['Position'].unique())]
    team_options = [{'label': 'All', 'value': 'all'}] + [{'label': team, 'value': team} for team in sorted(df['Team'].unique())]

    return [
        dbc.Row([
            dbc.Col(html.Div([html.Label("X-Axis:"), dcc.Dropdown(id='x-column', options=[{'label': col, 'value': col} for col in numeric_cols], value=filters.get('x_val', defaults['x_val']))]), width=12, md=6, lg=3),
            dbc.Col(html.Div([html.Label("Y-Axis:"), dcc.Dropdown(id='y-column', options=[{'label': col, 'value': col} for col in numeric_cols], value=filters.get('y_val', defaults['y_val']))]), width=12, md=6, lg=3),
            dbc.Col(html.Div([html.Label("Color By:"), dcc.Dropdown(id='color-column', options=all_cols_options, value=filters.get('color_val', defaults['color_val']), clearable=True)]), width=12, md=6, lg=3),
            dbc.Col(html.Div([html.Label("Size By:"), dcc.Dropdown(id='size-column', options=[{'label': col, 'value': col} for col in numeric_cols], value=filters.get('size_val', defaults['size_val']), clearable=True, placeholder="Select column for point size")]), width=12, md=6, lg=3),
        ], className="mb-3 g-2"), # g-2 for gutter space
        dbc.Row([
            dbc.Col(html.Div([html.Label("Filter by Name(s):"), dcc.Dropdown(id='name-dropdown', options=name_options, value=filters.get('name_val', defaults['name_val']), multi=True, placeholder="Select player(s)")]), width=12, md=4),
            dbc.Col(html.Div([html.Label("Filter by Position(s):"), dcc.Dropdown(id='position-dropdown', options=position_options, value=filters.get('pos_val', defaults['pos_val']), multi=True, placeholder="Select position(s)")]), width=12, md=4),
            dbc.Col(html.Div([html.Label("Filter by Team(s):"), dcc.Dropdown(id='team-dropdown', options=team_options, value=filters.get('team_val', defaults['team_val']), multi=True, placeholder="Select team(s)")]), width=12, md=4)
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col([
                dcc.Checklist(
                    id='toggle-text-labels',
                    options=[{'label': 'Toggle Text Labels', 'value': 'on'}],
                    value=filters.get('text_val', defaults['text_val']),
                    inline=True,
                    className="me-3"
                ),
                dcc.Checklist(
                    id='include-drafted-toggle',
                    options=[{'label': 'Include Drafted', 'value': 'include'}],
                    value=filters.get('drafted_val', defaults['drafted_val']),
                    inline=True,
                    className="me-3"
                ),
                dcc.Checklist(
                    id='exclude-undrafted-toggle',
                    options=[{'label': 'Exclude Undrafted', 'value': 'exclude'}],
                    value=filters.get('exclude_undrafted_val', defaults['exclude_undrafted_val']),
                    inline=True,
                    className="me-3"
                ),
                dcc.Checklist(
                    id='exclude-zeros-toggle',
                    options=[{'label': 'Exclude Zeros', 'value': 'exclude'}],
                    value=filters.get('exclude_zeros_val', defaults['exclude_zeros_val']),
                    inline=True,
                    className="me-3"
                ),
                dcc.Checklist(
                    id='average-by-color-toggle',
                    options=[{'label': 'Average on Color By', 'value': 'average'}],
                    value=filters.get('avg_by_color_val', defaults['avg_by_color_val']),
                    inline=True,
                    className="me-3"
                )
            ], className="d-flex flex-wrap")
        ])
    ]

@app.callback(
    Output('sliders-container', 'children'),
    Input('data-store', 'data'),
    Input('filters-store', 'data')
)
def update_sliders(json_data, filters):
    """Dynamically generates sliders in a grid layout, preserving their state."""
    if not json_data:
        return []

    filters = filters or {}
    slider_state = filters.get('sliders', {})

    df = pd.read_json(StringIO(json_data), orient='split')
    numeric_cols = sorted([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
    
    rows = []
    row_cols = []
    items_per_row = 3

    for i, col in enumerate(numeric_cols):
        min_val = df[col].min()
        max_val = df[col].max()
        
        if pd.isna(min_val) or pd.isna(max_val):
            min_val, max_val = 0, 1
        
        if min_val == max_val:
            max_val = min_val + 1
            
        current_min, current_max = slider_state.get(col, [min_val, max_val])
        step = (max_val - min_val) / 100 if min_val != max_val else 0.01

        slider_control = html.Div([
            dbc.Label(col),
            dbc.Row([
                dbc.Col(
                    dbc.Input(
                        id={'type': 'min-input', 'index': col}, type='number', value=f"{current_min:.2f}",
                        step=step, min=min_val, max=max_val, debounce=True,
                        size="sm", style={"fontSize": "0.875rem", "textAlign": "center"}
                    ), width=3
                ),
                dbc.Col(
                    dcc.RangeSlider(
                        id={'type': 'filter-slider', 'index': col}, min=min_val, max=max_val,
                        value=[current_min, current_max], step=step, marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ), width=6
                ),
                dbc.Col(
                    dbc.Input(
                        id={'type': 'max-input', 'index': col}, type='number', value=f"{current_max:.2f}",
                        step=step, min=min_val, max=max_val, debounce=True,
                        size="sm", style={"fontSize": "0.875rem", "textAlign": "center"}
                    ), width=3
                )
            ], align="center"),
            dbc.Row([
                dbc.Col(html.Small(f"Range: {min_val:.2f} to {max_val:.2f}", className="text-muted"))
            ], justify="center")
        ])

        col_item = dbc.Col(slider_control, width=12, lg=4, md=6)
        row_cols.append(col_item)

        if len(row_cols) == items_per_row or i == len(numeric_cols) - 1:
            rows.append(dbc.Row(row_cols, className="mb-3"))
            row_cols = []
    
    return rows

@app.callback(
    [Output({'type': 'filter-slider', 'index': dash.MATCH}, 'value'),
     Output({'type': 'min-input', 'index': dash.MATCH}, 'value'),
     Output({'type': 'max-input', 'index': dash.MATCH}, 'value')],
    [Input({'type': 'min-input', 'index': dash.MATCH}, 'value'),
     Input({'type': 'max-input', 'index': dash.MATCH}, 'value'),
     Input({'type': 'filter-slider', 'index': dash.MATCH}, 'value')],
    [State({'type': 'filter-slider', 'index': dash.MATCH}, 'min'),
     State({'type': 'filter-slider', 'index': dash.MATCH}, 'max')],
    prevent_initial_call=True
)
def sync_slider_inputs(min_input_val, max_input_val, slider_range, slider_min, slider_max):
    """
    Synchronize the min/max input boxes with the range slider.
    """
    triggered_id = ctx.triggered_id
    
    # Set default values from the slider's full range
    min_val, max_val = slider_min, slider_max
    
    # Determine which component triggered the callback
    if triggered_id.type == 'filter-slider':
        min_val, max_val = slider_range
    elif triggered_id.type == 'min-input':
        min_val = float(min_input_val)
        max_val = slider_range[1]
    elif triggered_id.type == 'max-input':
        min_val = slider_range[0]
        max_val = float(max_input_val)

    # Basic validation
    if min_val > max_val:
        min_val = max_val

    # Ensure values are within the slider's allowed range
    min_val = max(min_val, slider_min)
    max_val = min(max_val, slider_max)

    # Format for display
    min_input_display = f"{min_val:.2f}"
    max_input_display = f"{max_val:.2f}"
    
    return [min_val, max_val], min_input_display, max_input_display

@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('data-store', 'data'),
        Input('filters-store', 'data')
    ]
)
def update_scatter_plot(json_data, filters):
    if not json_data: return px.scatter(title="Loading data...")
    filters = filters or {}
    
    df = pd.read_json(StringIO(json_data), orient='split')
    
    # Handle 'Select All'
    name_val = filters.get('name_val', [])
    pos_val = filters.get('pos_val', [])
    team_val = filters.get('team_val', [])
    if 'all' in name_val: name_val = df['Name'].unique()
    if 'all' in pos_val: pos_val = df['Position'].unique()
    if 'all' in team_val: team_val = df['Team'].unique()

    # Apply filters
    if name_val: df = df[df['Name'].isin(name_val)]
    if pos_val: df = df[df['Position'].isin(pos_val)]
    if team_val: df = df[df['Team'].isin(team_val)]
    
    for col, value in filters.get('sliders', {}).items():
        if value:
           df = df[(df[col] >= value[0]) & (df[col] <= value[1])]

    if 'include' not in filters.get('drafted_val', []):
        df = df[df['Draft Status'] == 'Available']
    
    if 'exclude' in filters.get('exclude_undrafted_val', []):
        df = df[df['Draft Status'] == 'Drafted']
        
    x_col = filters.get('x_val', 'VORP')
    y_col = filters.get('y_val', 'Game Score')
    color_col = filters.get('color_val', 'Team')
    size_col = filters.get('size_val')

    if 'exclude' in filters.get('exclude_zeros_val', []):
        if x_col in df.columns and y_col in df.columns:
            df = df[(df[x_col] != 0) & (df[y_col] != 0)]

    if df.empty:
        return px.scatter(title="No data to display with current filter settings")

    # Check if aggregation is enabled
    avg_by_color = 'average' in filters.get('avg_by_color_val', [])

    if avg_by_color and color_col and color_col in df.columns:
        agg_cols = {}
        if x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
            agg_cols[x_col] = 'mean'
        if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            agg_cols[y_col] = 'mean'
        if size_col and size_col in df.columns and pd.api.types.is_numeric_dtype(df[size_col]):
            agg_cols[size_col] = 'mean'
        
        # Also get the count of players per group
        agg_cols['Name'] = 'count'

        plot_df = df.groupby(color_col).agg(agg_cols).reset_index()
        plot_df.rename(columns={'Name': 'Player Count'}, inplace=True)
        
        hover_data = [color_col, 'Player Count', x_col, y_col]
        if size_col in plot_df.columns: hover_data.append(size_col)
        
        text_labels = plot_df[color_col]
        trendline = None
    else:
        plot_df = df
        hover_data=['Name', 'Position', 'Team', 'Draft Status', 'Overall Rating']
        text_labels = plot_df['Name'] if 'on' in filters.get('text_val', []) else None
        trendline = 'ols'

    # Round all numeric columns for cleaner display in hover text and axes
    numeric_cols_in_df = plot_df.select_dtypes(include=np.number).columns
    plot_df[numeric_cols_in_df] = plot_df[numeric_cols_in_df].round(3)

    # Preprocess size and text
    size_data = preprocess_size_data(plot_df[size_col]) if size_col and size_col in plot_df.columns else None

    # Create plot
    # Create plot; if trendline backend (statsmodels/scipy) is unavailable in env, fallback without trendline
    try:
        fig = px.scatter(
            plot_df, x=x_col, y=y_col,
            color=color_col,
            size=size_data,
            text=text_labels,
            hover_data=hover_data,
            trendline=trendline,
        )
    except Exception:
        fig = px.scatter(
            plot_df, x=x_col, y=y_col,
            color=color_col,
            size=size_data,
            text=text_labels,
            hover_data=hover_data,
        )
    fig.update_traces(textposition='top center', textfont_size=8)
    return fig

@app.callback(
    [Output('ideal-team-container', 'children'),
     Output('draft-board-container', 'children')],
    [Input('generate-strategy-btn', 'n_clicks')],
    [State('data-store', 'data')]
)
def update_strategy_view(n_clicks, json_data):
    if n_clicks is None or not json_data:
        return None, None
        
    df = pd.read_json(StringIO(json_data), orient='split')
    available_players = df[df['Draft Status'] == 'Available']
    
    ideal_team, draft_board = get_draft_recommendations(available_players)
    
    if ideal_team.empty or draft_board.empty:
        return "Not enough players to generate a recommendation.", "Not enough players to generate a recommendation."

    # Round statistics for cleaner display
    for col in ['VORP', 'VONA']:
        if col in ideal_team.columns:
            ideal_team[col] = ideal_team[col].round(3)
        if col in draft_board.columns:
            draft_board[col] = draft_board[col].round(3)
    
    # --- Table Styling ---
    style_header = {
        'backgroundColor': 'rgb(30, 30, 30)',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'center'
    }
    style_cell = {
        'textAlign': 'center',
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white',
        'border': '1px solid rgb(80, 80, 80)'
    }
    style_cell_conditional = [
        {'if': {'column_id': 'Name'}, 'textAlign': 'left'}
    ]

    # Ideal Team Table
    ideal_team_num_cols = ['Overall Rating', 'VORP']
    ideal_team_styles = []
    for col in ideal_team_num_cols:
        ideal_team_styles.extend(generate_heatmap_style(ideal_team, col))

    ideal_team_table = dash_table.DataTable(
        data=ideal_team.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in ['Name', 'Position', 'Overall Rating', 'VORP']],
        style_header=style_header,
        style_cell=style_cell,
        style_cell_conditional=style_cell_conditional,
        style_data_conditional=ideal_team_styles,
    )
    
    # Draft Board Table
    draft_board_num_cols = ['Overall Rating', 'VORP', 'VONA']
    draft_board_styles = []
    for col in draft_board_num_cols:
        draft_board_styles.extend(generate_heatmap_style(draft_board, col))
    
    draft_board_table = dash_table.DataTable(
        data=draft_board.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in ['Name', 'Position', 'Overall Rating', 'VORP', 'VONA']],
        style_header=style_header,
        style_cell=style_cell,
        style_cell_conditional=style_cell_conditional,
        style_data_conditional=draft_board_styles,
    )
    
    return ideal_team_table, draft_board_table

# ================= Regular Season Callbacks =================

@app.callback(
    Output('rs-data-store', 'data'),
    [Input('rs-interval', 'n_intervals'), Input('rs-refresh-btn', 'n_clicks')]
)
def rs_update_data_store(n_intervals: int, n_clicks: Optional[int]) -> Dict[str, str]:
    data = get_regular_season_data()
    # Convert dataframes to JSON for storage
    payload = {k: v.to_json(date_format='iso', orient='split') for k, v in data.items()}
    return payload

@app.callback(
    [Output('rs-standings-table', 'children'), Output('rs-power-table', 'children')],
    Input('rs-data-store', 'data')
)
def rs_render_overview(json_payload: Optional[Dict[str, str]]) -> Tuple[Any, Any]:
    if not json_payload:
        return "Loading...", "Loading..."
    standings = pd.read_json(StringIO(json_payload['standings']), orient='split') if 'standings' in json_payload else pd.DataFrame()
    team_stats = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()

    def _table(df: DataFrame) -> Any:
        if df.empty:
            return "No data"
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        styles = []
        for col in num_cols:
            styles.extend(generate_heatmap_style(df, col))
        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_data_conditional=styles,
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'border': '1px solid rgb(80, 80, 80)'}
        )

    # Power rankings: combine standings with net metrics
    power_df = pd.DataFrame()
    if not standings.empty:
        power_df = standings.copy()
        # heuristic: blend win% and net rating/point differential when available
        win_pct_col = next((c for c in power_df.columns if c.lower() in {'percentage', 'win%', 'pct'}), None)
        pdiff_col = None
        nr_col = None
        if not team_stats.empty:
            # try common columns
            for c in team_stats.columns:
                lc = c.lower()
                if 'point differential' in lc:
                    pdiff_col = c
                if 'net rating' in lc:
                    nr_col = c
        # Join by Abbreviation or Team
        join_key = 'Abbreviation' if 'Abbreviation' in power_df.columns and 'Abbreviation' in team_stats.columns else 'Team'
        if not team_stats.empty and join_key in team_stats.columns and join_key in power_df.columns:
            merged = pd.merge(power_df, team_stats[[join_key] + [c for c in [pdiff_col, nr_col] if c]], on=join_key, how='left')
        else:
            merged = power_df
        # Scoring
        score = pd.Series(0.0, index=merged.index)
        if win_pct_col and win_pct_col in merged.columns:
            score = score + (merged[win_pct_col].fillna(0)) * 100.0
        if pdiff_col and pdiff_col in merged.columns:
            score = score + merged[pdiff_col].fillna(0) * 2.0
        if nr_col and nr_col in merged.columns:
            score = score + merged[nr_col].fillna(0) * 3.0
        merged['Power Score'] = score
        merged = merged.sort_values('Power Score', ascending=False)
        rank_col = 'Power Rank'
        merged[rank_col] = range(1, len(merged) + 1)
        # Tidy columns
        display_cols = [rank_col, 'Power Score'] + [c for c in merged.columns if c not in {rank_col, 'Power Score'}]
        power_df = merged[display_cols]

    return _table(standings), _table(power_df)

@app.callback(
    [Output('rs-team-x', 'options'), Output('rs-team-y', 'options'), Output('rs-team-color', 'options'), Output('rs-team-size', 'options'),
     Output('rs-player-x', 'options'), Output('rs-player-y', 'options'), Output('rs-player-color', 'options'), Output('rs-player-size', 'options')],
    Input('rs-data-store', 'data')
)
def rs_fill_dropdowns(json_payload: Optional[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    if not json_payload:
        empty = []
        return empty, empty, empty, empty, empty, empty, empty, empty
    team_stats = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()
    player_stats = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    def opts(df: DataFrame):
        if df.empty:
            return []
        return [{'label': c, 'value': c} for c in df.columns]
    return opts(team_stats), opts(team_stats), opts(team_stats), opts(team_stats), opts(player_stats), opts(player_stats), opts(player_stats), opts(player_stats)

@app.callback(
    Output('rs-team-graph', 'figure'),
    [Input('rs-data-store', 'data'), Input('rs-team-x', 'value'), Input('rs-team-y', 'value'), Input('rs-team-color', 'value'), Input('rs-team-size', 'value')]
)
def rs_team_plot(json_payload: Optional[Dict[str, str]], x: Optional[str], y: Optional[str], color: Optional[str], size: Optional[str]) -> Any:
    if not json_payload:
        return px.scatter(title='Loading...')
    df = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()
    if df.empty or not x or not y:
        return px.scatter(title='Select axes')
    size_series = preprocess_size_data(df[size]) if size and size in df.columns else None
    return px.scatter(df, x=x, y=y, color=color if color in df.columns else None, size=size_series, hover_data=[col for col in df.columns if col not in {x, y}])

@app.callback(
    Output('rs-player-graph', 'figure'),
    [Input('rs-data-store', 'data'), Input('rs-player-x', 'value'), Input('rs-player-y', 'value'), Input('rs-player-color', 'value'), Input('rs-player-size', 'value')]
)
def rs_player_plot(json_payload: Optional[Dict[str, str]], x: Optional[str], y: Optional[str], color: Optional[str], size: Optional[str]) -> Any:
    if not json_payload:
        return px.scatter(title='Loading...')
    df = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    if df.empty or not x or not y:
        return px.scatter(title='Select axes')
    size_series = preprocess_size_data(df[size]) if size and size in df.columns else None
    return px.scatter(df, x=x, y=y, color=color if color in df.columns else None, size=size_series, hover_data=[col for col in df.columns if col not in {x, y}])

@app.callback(
    Output('rs-top-players-table', 'children'),
    Input('rs-data-store', 'data')
)
def rs_top_players(json_payload: Optional[Dict[str, str]]) -> Any:
    if not json_payload:
        return "Loading..."
    df = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    if df.empty:
        return "No data"
    # Choose a ranking metric; fall back to Points then VORP then Game Score-like if present
    rank_metric = None
    for candidate in ['VORP', 'Game Score', 'Plus Minus', 'Value', 'Points']:
        if candidate in df.columns:
            rank_metric = candidate
            break
    if not rank_metric:
        rank_metric = df.select_dtypes(include=np.number).columns[:1][0] if not df.select_dtypes(include=np.number).empty else None
    if not rank_metric:
        return "No numeric columns to rank"
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    results = []
    pos_col = 'Position' if 'Position' in df.columns else None
    if not pos_col:
        # If no position, just take top N overall
        top_overall = df.sort_values(rank_metric, ascending=False).head(15)
        show_cols = [c for c in ['Name', rank_metric] if c in top_overall.columns]
        return dash_table.DataTable(data=top_overall[show_cols].to_dict('records'), columns=[{"name": c, "id": c} for c in show_cols])
    for pos in positions:
        top_pos = df[df[pos_col] == pos].sort_values(rank_metric, ascending=False).head(5)
        top_pos = top_pos.assign(Position=pos)
        results.append(top_pos)
    final_df = pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
    if final_df.empty:
        return "No data"
    show_cols = [c for c in ['Position', 'Name', rank_metric, 'Team'] if c in final_df.columns]
    styles = []
    if rank_metric in final_df.columns:
        styles = generate_heatmap_style(final_df, rank_metric)
    return dash_table.DataTable(data=final_df[show_cols].to_dict('records'), columns=[{"name": c, "id": c} for c in show_cols], style_data_conditional=styles)

@app.callback(
    Output('rs-trade-suggestions', 'children'),
    [Input('rs-trade-btn', 'n_clicks'), State('rs-team-select', 'value'), State('rs-data-store', 'data')]
)
def rs_trade_suggestions(n_clicks: Optional[int], team_value: Optional[str], json_payload: Optional[Dict[str, str]]) -> Any:
    if not n_clicks:
        return None
    if not json_payload or not team_value:
        return "Select a team first"
    players = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    teams = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()
    if players.empty or teams.empty:
        return "Missing player or team data"
    # Determine the selected team row
    team_row = pd.DataFrame()
    if 'Abbreviation' in teams.columns and team_value in teams['Abbreviation'].astype(str).values:
        team_row = teams[teams['Abbreviation'].astype(str) == str(team_value)].head(1)
    elif 'Team' in teams.columns and team_value in teams['Team'].astype(str).values:
        team_row = teams[teams['Team'].astype(str) == str(team_value)].head(1)
    if team_row.empty:
        return "Selected team not found in team statistics"
    # Identify league average row if available
    league_row = pd.DataFrame()
    if 'Team' in teams.columns:
        league_row = teams[teams['Team'].astype(str).str.contains('League', case=False, na=False)].head(1)
    if league_row.empty:
        league_row = teams.select_dtypes(include=np.number).mean(numeric_only=True).to_frame().T
    # Compute weaknesses using overlapping numeric stats between team and player stats
    team_num_cols = set(teams.select_dtypes(include=np.number).columns)
    player_num_cols = set(players.select_dtypes(include=np.number).columns)
    overlap = [c for c in team_num_cols.intersection(player_num_cols) if c not in {'Turnovers'}]
    if not overlap:
        return "No overlapping numeric stat columns to score"
    team_vals = team_row[overlap].iloc[0]
    league_vals = league_row[overlap].iloc[0]
    deltas = (team_vals - league_vals).sort_values()  # most negative are weak
    weak_stats = list(deltas.index[:3]) if len(deltas.index) >= 3 else list(deltas.index)
    # Score players by z-sum of weak stats
    candidates = players.copy()
    score = pd.Series(0.0, index=candidates.index)
    for stat in weak_stats:
        if stat in candidates.columns:
            mu = candidates[stat].mean()
            sd = candidates[stat].std()
            if sd and sd > 0:
                z = (candidates[stat] - mu) / sd
            else:
                z = 0
            score = score + z.fillna(0)
    candidates['Trade Score'] = score
    suggestions = candidates.sort_values('Trade Score', ascending=False).head(10)
    show_cols = [c for c in ['Name'] + weak_stats + ['Trade Score', 'Team', 'Position'] if c in suggestions.columns]
    return dash_table.DataTable(data=suggestions[show_cols].to_dict('records'), columns=[{"name": c, "id": c} for c in show_cols])

@app.callback(
    Output('rs-team-select', 'options'),
    Input('rs-data-store', 'data')
)
def rs_team_select_options(json_payload: Optional[Dict[str, str]]) -> List[Dict[str, str]]:
    if not json_payload:
        return []
    standings = pd.read_json(StringIO(json_payload['standings']), orient='split') if 'standings' in json_payload else pd.DataFrame()
    if standings.empty:
        return []
    label_col = 'Team' if 'Team' in standings.columns else 'Abbreviation'
    # Prefer using Abbreviation as value for stable lookups
    if 'Abbreviation' in standings.columns:
        return [{'label': str(row[label_col]), 'value': str(row['Abbreviation'])} for _, row in standings.iterrows()]
    return [{'label': str(row[label_col]), 'value': str(row[label_col])} for _, row in standings.iterrows()]

if __name__ == '__main__':
    app.run_server(debug=True) 