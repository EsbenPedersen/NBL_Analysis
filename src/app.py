import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, no_update, ctx
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import numpy as np
import pandas as pd
from io import StringIO

from src.data_loader import get_google_sheets_data
from src.data_processing import process_data
from src.draft_optimizer import get_draft_recommendations

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server
load_figure_template("cyborg")

# --- Helper Functions ---
def create_slider_marks(min_val, max_val, num_marks=10):
    """Create evenly spaced, rounded marks for a slider."""
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return {0: '0'}
    step_values = np.linspace(min_val, max_val, num_marks)
    return {int(value): '{:.2f}'.format(value) for value in step_values}

def preprocess_size_data(size_data):
    """Offsets and scales data for scatter plot point sizes."""
    if size_data.empty or size_data.isnull().all():
        return pd.Series(10, index=size_data.index)
    min_val = size_data.min()
    if min_val < 0:
        size_data = size_data + abs(min_val)
    max_val = size_data.max()
    if max_val is None or max_val == 0 or max_val == size_data.min():
        return pd.Series(10, index=size_data.index)
    size_data = 10 + ((size_data - size_data.min()) / (max_val - size_data.min())) * 40
    return size_data

# --- App Layout ---
app.layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Store(id='filters-store', data={}),
    dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0), # 60 seconds
    dbc.Row([
        dbc.Col(html.H1("NBL Draft Analysis"), width='auto'),
        dbc.Col(dbc.Button("Refresh Data", id="refresh-data-btn", color="secondary"), width='auto', className="align-self-center"),
        dbc.Col(dbc.Button("Reset Filters", id="reset-filters-btn", color="light"), width='auto', className="align-self-center")
    ], align="center", className="mb-3"),
    html.Div(id='controls-container'),
    dcc.Graph(id='scatter-plot'),
    html.Hr(),
    html.H2("Draft Strategy"),
    dbc.Button("Generate Draft Strategy", id="generate-strategy-btn", color="primary", className="mb-3"),
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Ideal Team (Top 2 per Position)"),
            html.Div(id='ideal-team-container')
        ]), width=6),
        dbc.Col(html.Div([
            html.H3("Prioritized Draft Board (by Scarcity)"),
            html.Div(id='draft-board-container')
        ]), width=6),
    ]),
    html.Hr(),
    html.H2("Player Filters"),
    html.Div(id='sliders-container')
])

# --- Callbacks ---
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
     Input('exclude-zeros-toggle', 'value'),
     Input({'type': 'filter-slider', 'index': dash.ALL}, 'value')],
    [State({'type': 'filter-slider', 'index': dash.ALL}, 'id'), State('filters-store', 'data')],
    prevent_initial_call=True
)
def update_filters_store(x, y, color, size, names, positions, teams, text, drafted, exclude_zeros, slider_values, slider_ids, current_filters):
    current_filters = current_filters or {}
    triggered_id = ctx.triggered_id
    
    if isinstance(triggered_id, dict) and triggered_id.get('type') == 'filter-slider':
        slider_state = {s_id['index']: s_val for s_id, s_val in zip(slider_ids, slider_values)}
        current_filters['sliders'] = slider_state
    else:
        current_filters.update({
            'x_val': x, 'y_val': y, 'color_val': color, 'size_val': size,
            'name_val': names, 'pos_val': positions, 'team_val': teams,
            'text_val': text, 'drafted_val': drafted, 'exclude_zeros_val': exclude_zeros
        })
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
    
    defaults = {'x_val': 'Overall Rating', 'y_val': 'Points', 'color_val': 'Position', 'size_val': 'Minutes',
                'name_val': [], 'pos_val': [], 'team_val': [], 'text_val': [], 'drafted_val': [], 'exclude_zeros_val': []}
    
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
            dbc.Col(dcc.Checklist(id='exclude-zeros-toggle', options=[{'label': 'Exclude Zeros', 'value': 'exclude'}], value=filters.get('exclude_zeros_val', defaults['exclude_zeros_val']), labelStyle={'display': 'inline-block'}), width={'size': 'auto'}),
        ], className="mt-2", justify="center"),
    ]

@app.callback(
    Output('sliders-container', 'children'),
    Input('data-store', 'data'),
    Input('filters-store', 'data')
)
def update_sliders(json_data, filters):
    """Dynamically generates sliders, preserving their state."""
    if not json_data:
        return []
    
    filters = filters or {}
    slider_state = filters.get('sliders', {})
        
    df = pd.read_json(StringIO(json_data), orient='split')
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    return [html.Div([
        html.Label(col),
        dcc.RangeSlider(
            id={'type': 'filter-slider', 'index': col},
            min=df[col].min(),
            max=df[col].max(),
            value=slider_state.get(col, [df[col].min(), df[col].max()]),
            step=(df[col].max() - df[col].min()) / 100 if df[col].min() != df[col].max() else 1,
            marks=create_slider_marks(df[col].min(), df[col].max())
        )
    ]) for col in numeric_cols]

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
        
    x_col = filters.get('x_val', 'Overall Rating')
    y_col = filters.get('y_val', 'Points')
    
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

    # Round statistics for cleaner display in tables
    for col in ['Value', 'VONA']:
        if col in ideal_team.columns:
            ideal_team[col] = ideal_team[col].round(3)
        if col in draft_board.columns:
            draft_board[col] = draft_board[col].round(3)
    
    ideal_team_table = dash_table.DataTable(
        data=ideal_team.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in ['Name', 'Position', 'Value']],
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        },
    )
    
    draft_board_table = dash_table.DataTable(
        data=draft_board.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in ['Name', 'Position', 'Value', 'VONA']],
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        },
    )
    
    return ideal_team_table, draft_board_table

if __name__ == '__main__':
    app.run_server(debug=True) 