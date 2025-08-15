import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, no_update, ctx
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from src.data_loader import get_google_sheets_data
from src.data_processing import process_data
from src.draft_optimizer import get_draft_recommendations
from src.ui_helpers import generate_heatmap_style, preprocess_size_data
import logging


dash.register_page(__name__, path="/", name="Draft")


layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Store(id='filters-store', data={}),
    dcc.Interval(id='interval-component', interval=5 * 60 * 1000, n_intervals=0),

    dbc.Row([
        dbc.Col(html.H1("NBL Draft Analysis"), width='auto'),
        dbc.Col(dbc.Button("Refresh Data", id="refresh-data-btn", color="secondary"), width='auto', className="align-self-center"),
        dbc.Col(dbc.Button("Reset Filters", id="reset-filters-btn", color="light"), width='auto', className="align-self-center")
    ], align="center", className="mb-4"),

    dbc.Card(
        dbc.CardBody([
            html.Div(id='controls-container'),
            dcc.Graph(id='scatter-plot', style={'height': '60vh'}),
        ]),
        className="mb-4",
    ),

    dbc.Card([
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
    ], className="mb-4"),

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


@dash.callback(
    Output("filters-collapse", "is_open"),
    [Input("toggle-filters-btn", "n_clicks")],
    [State("filters-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_filters_collapse(n: Optional[int], is_open: Optional[bool]) -> Optional[bool]:
    if n:
        return not is_open
    return is_open


@dash.callback(
    Output('data-store', 'data'),
    [Input('interval-component', 'n_intervals'), Input('refresh-data-btn', 'n_clicks')]
)
def update_data_store(n_intervals: int, n_clicks: Optional[int]) -> str:
    try:
        raw_data = get_google_sheets_data()
        logging.info("Draft sheets fetched %d tabs", len(raw_data))
        processed_data = process_data(raw_data)
        for col in processed_data.columns:
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].fillna(0)
        return processed_data.to_json(date_format='iso', orient='split')
    except Exception:
        # Gracefully handle missing credentials or network issues
        logging.warning("Draft data fetch failed; returning empty payload", exc_info=True)
        return ""


@dash.callback(
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

    slider_map = {s_id['index']: s_val for s_id, s_val in zip(slider_ids, slider_values)}

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


@dash.callback(
    Output('filters-store', 'data', allow_duplicate=True),
    Input('reset-filters-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters_store(n_clicks):
    return {}


@dash.callback(
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


@dash.callback(
    Output('controls-container', 'children'),
    [Input('data-store', 'data'), Input('filters-store', 'data')]
)
def update_controls(json_data, filters):
    if not json_data:
        return []
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
        ], className="mb-3 g-2"),
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


@dash.callback(
    Output('sliders-container', 'children'),
    Input('data-store', 'data'),
    Input('filters-store', 'data')
)
def update_sliders(json_data, filters):
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


@dash.callback(
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
    triggered_id = ctx.triggered_id
    min_val, max_val = slider_min, slider_max
    if triggered_id.type == 'filter-slider':
        min_val, max_val = slider_range
    elif triggered_id.type == 'min-input':
        min_val = float(min_input_val)
        max_val = slider_range[1]
    elif triggered_id.type == 'max-input':
        min_val = slider_range[0]
        max_val = float(max_input_val)
    if min_val > max_val:
        min_val = max_val
    min_val = max(min_val, slider_min)
    max_val = min(max_val, slider_max)
    return [min_val, max_val], f"{min_val:.2f}", f"{max_val:.2f}"


@dash.callback(
    Output('scatter-plot', 'figure'),
    [Input('data-store', 'data'), Input('filters-store', 'data')]
)
def update_scatter_plot(json_data, filters):
    if not json_data:
        return px.scatter(title="Loading data...")
    filters = filters or {}
    df = pd.read_json(StringIO(json_data), orient='split')

    name_val = filters.get('name_val', [])
    pos_val = filters.get('pos_val', [])
    team_val = filters.get('team_val', [])
    if 'all' in name_val:
        name_val = df['Name'].unique()
    if 'all' in pos_val:
        pos_val = df['Position'].unique()
    if 'all' in team_val:
        team_val = df['Team'].unique()

    if name_val:
        df = df[df['Name'].isin(name_val)]
    if pos_val:
        df = df[df['Position'].isin(pos_val)]
    if team_val:
        df = df[df['Team'].isin(team_val)]

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

    avg_by_color = 'average' in filters.get('avg_by_color_val', [])

    if avg_by_color and color_col and color_col in df.columns:
        agg_cols: Dict[str, str] = {}
        if x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
            agg_cols[x_col] = 'mean'
        if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            agg_cols[y_col] = 'mean'
        if size_col and size_col in df.columns and pd.api.types.is_numeric_dtype(df[size_col]):
            agg_cols[size_col] = 'mean'
        agg_cols['Name'] = 'count'
        plot_df = df.groupby(color_col).agg(agg_cols).reset_index()
        plot_df.rename(columns={'Name': 'Player Count'}, inplace=True)
        hover_data = [color_col, 'Player Count', x_col, y_col]
        if size_col in plot_df.columns:
            hover_data.append(size_col)
        text_labels = plot_df[color_col]
        trendline = None
    else:
        plot_df = df
        hover_data = ['Name', 'Position', 'Team', 'Draft Status', 'Overall Rating']
        text_labels = plot_df['Name'] if 'on' in filters.get('text_val', []) else None
        trendline = 'ols'

    numeric_cols_in_df = plot_df.select_dtypes(include='number').columns
    plot_df[numeric_cols_in_df] = plot_df[numeric_cols_in_df].round(3)

    size_data = preprocess_size_data(plot_df[size_col]) if size_col and size_col in plot_df.columns else None

    try:
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, size=size_data, text=text_labels, hover_data=hover_data, trendline=trendline)
    except Exception:
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, size=size_data, text=text_labels, hover_data=hover_data)
    fig.update_traces(textposition='top center', textfont_size=8)
    return fig


@dash.callback(
    [Output('ideal-team-container', 'children'), Output('draft-board-container', 'children')],
    [Input('generate-strategy-btn', 'n_clicks')],
    [State('data-store', 'data')]
)
def update_strategy_view(n_clicks: Optional[int], json_data: Optional[str]):
    if n_clicks is None or not json_data:
        return None, None
    df = pd.read_json(StringIO(json_data), orient='split')
    available_players = df[df['Draft Status'] == 'Available']
    ideal_team, draft_board = get_draft_recommendations(available_players)
    if ideal_team.empty or draft_board.empty:
        return "Not enough players to generate a recommendation.", "Not enough players to generate a recommendation."

    for col in ['VORP', 'VONA']:
        if col in ideal_team.columns:
            ideal_team[col] = ideal_team[col].round(3)
        if col in draft_board.columns:
            draft_board[col] = draft_board[col].round(3)

    style_header = {'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
    style_cell = {'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'border': '1px solid rgb(80, 80, 80)'}
    style_cell_conditional = [{'if': {'column_id': 'Name'}, 'textAlign': 'left'}]

    ideal_team_styles: List[Dict[str, object]] = []
    for col in ['Overall Rating', 'VORP']:
        ideal_team_styles.extend(generate_heatmap_style(ideal_team, col))

    ideal_team_table = dash_table.DataTable(
        data=ideal_team.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in ['Name', 'Position', 'Overall Rating', 'VORP']],
        style_header=style_header,
        style_cell=style_cell,
        style_cell_conditional=style_cell_conditional,
        style_data_conditional=ideal_team_styles,
    )

    draft_board_styles: List[Dict[str, object]] = []
    for col in ['Overall Rating', 'VORP', 'VONA']:
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


