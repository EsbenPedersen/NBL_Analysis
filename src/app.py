import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, no_update, ctx
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import numpy as np
import pandas as pd
from pandas import DataFrame
from io import StringIO

from src.data_loader import get_google_sheets_data
from src.data_processing import process_data
from src.draft_optimizer import get_draft_recommendations

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
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

# --- App Layout ---
app.layout = dbc.Container([
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
        dbc.Collapse(
            dbc.CardBody(html.Div(id='sliders-container')),
            id="filters-collapse",
            is_open=False,
        ),
    ]),

], fluid=True, className="py-4 px-5")

# --- Callbacks ---
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
     Input({'type': 'filter-slider', 'index': dash.ALL}, 'value'),
     Input({'type': 'min-input', 'index': dash.ALL}, 'value'),
     Input({'type': 'max-input', 'index': dash.ALL}, 'value')],
    [State({'type': 'filter-slider', 'index': dash.ALL}, 'id'),
     State({'type': 'min-input', 'index': dash.ALL}, 'id'),
     State({'type': 'max-input', 'index': dash.ALL}, 'id'),
     State('filters-store', 'data')],
    prevent_initial_call=True
)
def update_filters_store(x, y, color, size, names, positions, teams, text, drafted, exclude_undrafted, exclude_zeros,
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
            'text_val': text, 'drafted_val': drafted, 'exclude_undrafted_val': exclude_undrafted, 'exclude_zeros_val': exclude_zeros
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
    Output('controls-container', 'children'),
    [Input('data-store', 'data'), Input('filters-store', 'data')]
)
def update_controls(json_data, filters):
    if not json_data: return []
    filters = filters or {}
    
    defaults = {'x_val': 'VORP', 'y_val': 'Game Score', 'color_val': 'A/TO', 'size_val': 'PPM',
                'name_val': [], 'pos_val': [], 'team_val': [], 'text_val': [], 'drafted_val': [], 'exclude_undrafted_val': [], 'exclude_zeros_val': ['exclude']}
    
    df = pd.read_json(StringIO(json_data), orient='split')
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    all_cols_options = [{'label': col, 'value': col} for col in df.columns]
    name_options = [{'label': 'All', 'value': 'all'}] + [{'label': name, 'value': name} for name in sorted(df['Name'].unique())]
    position_options = [{'label': 'All', 'value': 'all'}] + [{'label': pos, 'value': pos} for pos in sorted(df['Position'].unique())]
    team_options = [{'label': 'All', 'value': 'all'}] + [{'label': team, 'value': team} for team in sorted(df['Team'].unique())]

    return [
        dbc.Row([
            dbc.Col(html.Div([html.Label("X-Axis:"), dcc.Dropdown(id='x-column', options=[{'label': col, 'value': col} for col in numeric_cols], value=filters.get('x_val', defaults['x_val']))]), width=3),
            dbc.Col(html.Div([html.Label("Y-Axis:"), dcc.Dropdown(id='y-column', options=[{'label': col, 'value': col} for col in numeric_cols], value=filters.get('y_val', defaults['y_val']))]), width=3),
            dbc.Col(html.Div([html.Label("Color By:"), dcc.Dropdown(id='color-column', options=all_cols_options, value=filters.get('color_val', defaults['color_val']), clearable=True)]), width=3),
            dbc.Col(html.Div([html.Label("Size By:"), dcc.Dropdown(id='size-column', options=[{'label': col, 'value': col} for col in numeric_cols], value=filters.get('size_val', defaults['size_val']), clearable=True, placeholder="Select column for point size")]), width=3),
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='name-dropdown', options=name_options, value=filters.get('name_val', defaults['name_val']), multi=True, placeholder="Filter by Name(s)"), width=4),
            dbc.Col(dcc.Dropdown(id='position-dropdown', options=position_options, value=filters.get('pos_val', defaults['pos_val']), multi=True, placeholder="Filter by Position(s)"), width=4),
            dbc.Col(dcc.Dropdown(id='team-dropdown', options=team_options, value=filters.get('team_val', defaults['team_val']), multi=True, placeholder="Filter by Team(s)"), width=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Checklist(id='toggle-text-labels', options=[{'label': 'Toggle Text Labels', 'value': 'on'}], value=filters.get('text_val', defaults['text_val']), labelStyle={'display': 'inline-block'}), width={'size': 'auto'}),
            dbc.Col(dcc.Checklist(id='include-drafted-toggle', options=[{'label': 'Include Drafted', 'value': 'include'}], value=filters.get('drafted_val', defaults['drafted_val']), labelStyle={'display': 'inline-block'}), width={'size': 'auto'}),
            dbc.Col(dcc.Checklist(id='exclude-undrafted-toggle', options=[{'label': 'Exclude Undrafted', 'value': 'exclude'}], value=filters.get('exclude_undrafted_val', defaults['exclude_undrafted_val']), labelStyle={'display': 'inline-block'}), width={'size': 'auto'}),
            dbc.Col(dcc.Checklist(id='exclude-zeros-toggle', options=[{'label': 'Exclude Zeros', 'value': 'exclude'}], value=filters.get('exclude_zeros_val', defaults['exclude_zeros_val']), labelStyle={'display': 'inline-block'}), width={'size': 'auto'}),
        ], className="mt-2", justify="center"),
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
    [State({'type': 'min-input', 'index': dash.MATCH}, 'id'),
     State('data-store', 'data')],
    prevent_initial_call=True
)
def sync_slider_inputs(min_input, max_input, slider_value, input_id, json_data):
    """Synchronize slider and input values."""
    if not json_data:
        return dash.no_update, dash.no_update, dash.no_update
    
    triggered = ctx.triggered_id
    df = pd.read_json(StringIO(json_data), orient='split')
    col = input_id['index']
    
    if col not in df.columns:
        return dash.no_update, dash.no_update, dash.no_update
    
    min_val = df[col].min()
    max_val = df[col].max()
    
    if pd.isna(min_val) or pd.isna(max_val):
        min_val, max_val = 0, 1
    
    if min_val == max_val:
        max_val = min_val + 1
    
    # Convert inputs to floats, handling None/empty values
    try:
        min_input = float(min_input) if min_input is not None and min_input != '' else min_val
        max_input = float(max_input) if max_input is not None and max_input != '' else max_val
    except (ValueError, TypeError):
        min_input = min_val
        max_input = max_val
    
    # Ensure bounds are respected
    min_input = max(min_val, min_input)
    max_input = min(max_val, max_input)
    
    # Ensure min <= max
    if min_input > max_input:
        if triggered and triggered.get('type') == 'min-input':
            max_input = min_input
        else:
            min_input = max_input
    
    return [min_input, max_input], f"{min_input:.2f}", f"{max_input:.2f}"

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
    
    if 'exclude' in filters.get('exclude_zeros_val', []):
        if x_col in df.columns and y_col in df.columns:
            df = df[(df[x_col] != 0) & (df[y_col] != 0)]

    if df.empty:
        return px.scatter(title="No data to display with current filter settings")

    # Round all numeric columns for cleaner display in hover text and axes
    numeric_cols_in_df = df.select_dtypes(include=np.number).columns
    df[numeric_cols_in_df] = df[numeric_cols_in_df].round(3)

    # Preprocess size and text
    size_col = filters.get('size_val')
    size_data = preprocess_size_data(df[size_col]) if size_col and size_col in df.columns else None
    text_labels = df['Name'] if 'on' in filters.get('text_val', []) else None

    # Create plot
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=filters.get('color_val'),
        size=size_data,
        text=text_labels,
        hover_data=['Name', 'Position', 'Draft Status', 'Overall Rating'],
        trendline='ols',
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

if __name__ == '__main__':
    app.run_server(debug=True) 