import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, dash_table
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from src.data_loader import get_regular_season_data
from src.ui_helpers import generate_heatmap_style, preprocess_size_data
import logging


dash.register_page(__name__, path="/regular", name="Regular Season")

STANDINGS_DISPLAY_COLUMNS: List[str] = [
    'Team', 'General Manager', 'Wins', 'Losses', 'Percentage',
]

POWER_DISPLAY_COLUMNS: List[str] = [
    'Team', 'General Manager', 'Power Rank', 'Power Score', 'Point Differential', 'Net Rating'
]


layout = html.Div([
    dcc.Store(id='rs-data-store'),
    dcc.Interval(id='rs-interval', interval=60 * 1000, n_intervals=0),
    dbc.Row([
        dbc.Col(html.H1("Regular Season"), width='auto'),
        dbc.Col(dbc.Button("Refresh Season Data", id="rs-refresh-btn", color="secondary"), width='auto', className="align-self-center"),
    ], align="center", className="mb-4"),

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

    dbc.Card([
        dbc.CardHeader(html.H3("Top Players")),
        dbc.CardBody(html.Div(id='rs-top-players-table'))
    ], className="mb-4"),

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
                dbc.Row([
                    dbc.Col(
                        dcc.Checklist(
                            id='rs-team-text-toggle',
                            options=[{'label': 'Show GM Labels', 'value': 'on'}],
                            value=['on'],
                            inline=True,
                        ),
                        md=12
                    )
                ], className='mb-2'),
                dcc.Graph(id='rs-team-graph', style={'height': '50vh', 'width': '100%'}, config={'responsive': True})
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
                dbc.Row([
                    dbc.Col(
                        dcc.Checklist(
                            id='rs-player-text-toggle',
                            options=[{'label': 'Show Name Labels', 'value': 'on'}],
                            value=[],
                            inline=True,
                        ),
                        md=12
                    )
                ], className='mb-2'),
                dcc.Graph(id='rs-player-graph', style={'height': '50vh', 'width': '100%'}, config={'responsive': True})
            ])
        ]), md=6),
    ], className="mb-4"),
])


@dash.callback(
    Output('rs-data-store', 'data'),
    [Input('rs-interval', 'n_intervals'), Input('rs-refresh-btn', 'n_clicks')]
)
def rs_update_data_store(n_intervals: int, n_clicks: Optional[int]) -> Dict[str, str]:
    try:
        data = get_regular_season_data()
        payload = {k: v.to_json(date_format='iso', orient='split') for k, v in data.items()}
        logging.info("Regular season data fetched: standings=%s, team_stats=%s, player_stats=%s",
                     data.get('standings', pd.DataFrame()).shape,
                     data.get('team_stats', pd.DataFrame()).shape,
                     data.get('player_stats', pd.DataFrame()).shape)
        return payload
    except Exception:
        logging.warning("Regular season data fetch failed; returning empty payload", exc_info=True)
        return {}


@dash.callback(
    [Output('rs-standings-table', 'children'), Output('rs-power-table', 'children')],
    Input('rs-data-store', 'data')
)
def rs_render_overview(json_payload: Optional[Dict[str, str]]) -> Tuple[Any, Any]:
    if not json_payload:
        return "Loading...", "Loading..."
    standings = pd.read_json(StringIO(json_payload['standings']), orient='split') if 'standings' in json_payload else pd.DataFrame()
    team_stats = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()

    def _table(df: pd.DataFrame) -> Any:
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

    power_df = pd.DataFrame()
    if not standings.empty:
        power_df = standings.copy()
        win_pct_col = next((c for c in power_df.columns if c.lower() in {'percentage', 'win%', 'pct'}), None)
        pdiff_col = None
        nr_col = None
        if not team_stats.empty:
            for c in team_stats.columns:
                lc = c.lower()
                if 'point differential' in lc:
                    pdiff_col = c
                if 'net rating' in lc:
                    nr_col = c
        join_key = 'Abbreviation' if 'Abbreviation' in power_df.columns and 'Abbreviation' in team_stats.columns else 'Team'
        if not team_stats.empty and join_key in team_stats.columns and join_key in power_df.columns:
            merged = pd.merge(power_df, team_stats[[join_key] + [c for c in [pdiff_col, nr_col] if c]], on=join_key, how='left')
        else:
            merged = power_df
        score = pd.Series(0.0, index=merged.index)
        if win_pct_col and win_pct_col in merged.columns:
            score = score + (merged[win_pct_col].fillna(0)) * 100.0
        if pdiff_col and pdiff_col in merged.columns:
            score = score + merged[pdiff_col].fillna(0) * 2.0
        if nr_col and nr_col in merged.columns:
            score = score + merged[nr_col].fillna(0) * 3.0
        merged['Power Score'] = score.round(2)
        merged = merged.sort_values('Power Score', ascending=False)
        rank_col = 'Power Rank'
        merged[rank_col] = range(1, len(merged) + 1)
        default_cols = [rank_col, 'Power Score'] + [c for c in merged.columns if c not in {rank_col, 'Power Score'}]
        desired = [c for c in POWER_DISPLAY_COLUMNS if c in merged.columns]
        display_cols = desired if desired else default_cols
        power_df = merged[display_cols]

    # Trim standings columns for display using configured list, with fallback to all columns
    if not standings.empty:
        standings_cols = [c for c in STANDINGS_DISPLAY_COLUMNS if c in standings.columns]
        standings_df = standings[standings_cols] if standings_cols else standings
    else:
        standings_df = standings

    return _table(standings_df), _table(power_df)


@dash.callback(
    [Output('rs-team-x', 'options'), Output('rs-team-y', 'options'), Output('rs-team-color', 'options'), Output('rs-team-size', 'options'),
     Output('rs-player-x', 'options'), Output('rs-player-y', 'options'), Output('rs-player-color', 'options'), Output('rs-player-size', 'options')],
    Input('rs-data-store', 'data')
)
def rs_fill_dropdowns(json_payload: Optional[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    if not json_payload:
        empty: List[Dict[str, str]] = []
        return empty, empty, empty, empty, empty, empty, empty, empty
    team_stats = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()
    player_stats = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()

    def opts(df: pd.DataFrame) -> List[Dict[str, str]]:
        if df.empty:
            return []
        return [{'label': c, 'value': c} for c in df.columns]

    return opts(team_stats), opts(team_stats), opts(team_stats), opts(team_stats), opts(player_stats), opts(player_stats), opts(player_stats), opts(player_stats)


@dash.callback(
    [Output('rs-team-x', 'value'), Output('rs-team-y', 'value'), Output('rs-team-color', 'value'), Output('rs-team-size', 'value')],
    [Input('rs-team-x', 'options'), Input('rs-team-y', 'options'), Input('rs-team-color', 'options'), Input('rs-team-size', 'options')],
)
def rs_set_team_defaults(x_opts, y_opts, color_opts, size_opts):
    x_default = 'Point Differential'
    y_default = 'Net Rating'
    color_default = 'General Manager'
    size_default = 'Points Per Possession'
    def pick(opts, want):
        if not opts:
            return None
        values = {o['value'] for o in opts}
        return want if want in values else None
    return pick(x_opts, x_default), pick(y_opts, y_default), pick(color_opts, color_default), pick(size_opts, size_default)


@dash.callback(
    [Output('rs-player-x', 'value'), Output('rs-player-y', 'value'), Output('rs-player-color', 'value'), Output('rs-player-size', 'value')],
    [Input('rs-player-x', 'options'), Input('rs-player-y', 'options'), Input('rs-player-color', 'options'), Input('rs-player-size', 'options')],
)
def rs_set_player_defaults(x_opts, y_opts, color_opts, size_opts):
    x_default = 'VORP'
    y_default = 'Game Score'
    color_default = 'Position'
    size_default = None
    def pick(opts, want):
        if not opts or want is None:
            return None
        values = {o['value'] for o in opts}
        return want if want in values else None
    return pick(x_opts, x_default), pick(y_opts, y_default), pick(color_opts, color_default), size_default


@dash.callback(
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
    fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, size=size_series,
                     hover_name=df['Team'] if 'Team' in df.columns else None,
                     hover_data=[col for col in df.columns if col not in {x, y}])
    fig.update_layout(hovermode='closest')
    return fig


@dash.callback(
    Output('rs-player-graph', 'figure'),
    [Input('rs-data-store', 'data'), Input('rs-player-x', 'value'), Input('rs-player-y', 'value'), Input('rs-player-color', 'value'), Input('rs-player-size', 'value')]
)
def rs_player_plot(json_payload: Optional[Dict[str, str]], x: Optional[str], y: Optional[str], color: Optional[str], size: Optional[str]) -> Any:
    if not json_payload:
        return px.scatter(title='Loading...')
    # Use aggregated Player Statistics for player-level stats
    df = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    if df.empty or not x or not y:
        return px.scatter(title='Select axes')
    size_series = preprocess_size_data(df[size]) if size and size in df.columns else None
    fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, size=size_series,
                     hover_name=df['Name'] if 'Name' in df.columns else None,
                     hover_data=[col for col in df.columns if col not in {x, y}])
    fig.update_layout(hovermode='closest')
    return fig


@dash.callback(
    Output('rs-team-graph', 'figure', allow_duplicate=True),
    [Input('rs-team-text-toggle', 'value'), Input('rs-data-store', 'data'), Input('rs-team-x', 'value'), Input('rs-team-y', 'value'), Input('rs-team-color', 'value'), Input('rs-team-size', 'value')],
    prevent_initial_call=True
)
def rs_team_text_labels(toggle, json_payload, x, y, color, size):
    fig = rs_team_plot(json_payload, x, y, color, size)
    show = bool(toggle and 'on' in toggle)
    df = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if json_payload and 'team_stats' in json_payload else pd.DataFrame()
    if show and not df.empty and 'General Manager' in df.columns:
        fig.update_traces(text=df['General Manager'], textposition='top center', textfont_size=10)
    else:
        fig.update_traces(text=None)
    return fig


@dash.callback(
    Output('rs-player-graph', 'figure', allow_duplicate=True),
    [Input('rs-player-text-toggle', 'value'), Input('rs-data-store', 'data'), Input('rs-player-x', 'value'), Input('rs-player-y', 'value'), Input('rs-player-color', 'value'), Input('rs-player-size', 'value')],
    prevent_initial_call=True
)
def rs_player_text_labels(toggle, json_payload, x, y, color, size):
    fig = rs_player_plot(json_payload, x, y, color, size)
    show = bool(toggle and 'on' in toggle)
    df = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if json_payload and 'player_stats' in json_payload else pd.DataFrame()
    if show and not df.empty and 'Name' in df.columns:
        fig.update_traces(text=df['Name'], textposition='top center', textfont_size=10)
    else:
        fig.update_traces(text=None)
    return fig


@dash.callback(
    Output('rs-top-players-table', 'children'),
    Input('rs-data-store', 'data')
)
def rs_top_players(json_payload: Optional[Dict[str, str]]):
    if not json_payload:
        return "Loading..."
    df = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    if df.empty:
        return "No data"
    pos_col = 'Position' if 'Position' in df.columns else None
    if not pos_col:
        # Fallback: show overall top 5 with VORP/Plus Minus if available
        sort_by = 'VORP' if 'VORP' in df.columns else ('Plus Minus' if 'Plus Minus' in df.columns else None)
        if not sort_by:
            nums = df.select_dtypes(include='number')
            sort_by = nums.columns[:1][0] if not nums.empty else None
        if not sort_by:
            return "No numeric columns to rank"
        top_overall = df.sort_values(sort_by, ascending=False).head(5)
        show_cols = [c for c in ['Name', 'Team', 'VORP', 'Plus Minus'] if c in top_overall.columns]
        return dash_table.DataTable(data=top_overall[show_cols].to_dict('records'), columns=[{"name": c, "id": c} for c in show_cols])

    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    tables: List[Any] = []
    for pos in positions:
        pos_df = df[df[pos_col] == pos].copy()
        if pos_df.empty:
            continue
        # Rank by VORP if available; otherwise by Plus Minus; otherwise by first numeric
        sort_by = 'VORP' if 'VORP' in pos_df.columns else ('Plus Minus' if 'Plus Minus' in pos_df.columns else None)
        if not sort_by:
            nums = pos_df.select_dtypes(include='number')
            sort_by = nums.columns[:1][0] if not nums.empty else None
        if not sort_by:
            continue
        top_pos = pos_df.sort_values(sort_by, ascending=False).head(5)
        # Columns to display: Name, Team, VORP, Plus Minus (subset to available)
        show_cols = [c for c in ['Name', 'Team', 'VORP', 'Plus Minus'] if c in top_pos.columns]
        # Optional rounding for readability
        for col in ['VORP', 'Plus Minus']:
            if col in top_pos.columns:
                top_pos[col] = pd.to_numeric(top_pos[col], errors='coerce').round(2)
        # Add heatmap styling on VORP and Plus Minus if present
        styles: List[Dict[str, object]] = []
        for hcol in ['VORP', 'Plus Minus']:
            if hcol in top_pos.columns:
                styles.extend(generate_heatmap_style(top_pos, hcol))

        table = html.Div([
            html.H5(pos, className="mb-2"),
            dash_table.DataTable(
                data=top_pos[show_cols].to_dict('records'),
                columns=[{"name": c, "id": c} for c in show_cols],
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'border': '1px solid rgb(80, 80, 80)'},
                style_data_conditional=styles,
            )
        ], className="mb-3")
        tables.append(table)

    if not tables:
        return "No data"
    return tables


