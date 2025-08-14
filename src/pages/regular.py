import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, dash_table
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from src.data_loader import get_regular_season_data
from src.ui_helpers import generate_heatmap_style, preprocess_size_data


dash.register_page(__name__, path="/regular", name="Regular Season")


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
])


@dash.callback(
    Output('rs-data-store', 'data'),
    [Input('rs-interval', 'n_intervals'), Input('rs-refresh-btn', 'n_clicks')]
)
def rs_update_data_store(n_intervals: int, n_clicks: Optional[int]) -> Dict[str, str]:
    data = get_regular_season_data()
    payload = {k: v.to_json(date_format='iso', orient='split') for k, v in data.items()}
    return payload


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
        merged['Power Score'] = score
        merged = merged.sort_values('Power Score', ascending=False)
        rank_col = 'Power Rank'
        merged[rank_col] = range(1, len(merged) + 1)
        display_cols = [rank_col, 'Power Score'] + [c for c in merged.columns if c not in {rank_col, 'Power Score'}]
        power_df = merged[display_cols]

    return _table(standings), _table(power_df)


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


@dash.callback(
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
    rank_metric: Optional[str] = None
    for candidate in ['VORP', 'Game Score', 'Plus Minus', 'Value', 'Points']:
        if candidate in df.columns:
            rank_metric = candidate
            break
    if not rank_metric:
        nums = df.select_dtypes(include='number')
        rank_metric = nums.columns[:1][0] if not nums.empty else None
    if not rank_metric:
        return "No numeric columns to rank"
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    results = []
    pos_col = 'Position' if 'Position' in df.columns else None
    if not pos_col:
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
    styles: List[Dict[str, object]] = []
    if rank_metric in final_df.columns:
        styles = generate_heatmap_style(final_df, rank_metric)
    return dash_table.DataTable(data=final_df[show_cols].to_dict('records'), columns=[{"name": c, "id": c} for c in show_cols], style_data_conditional=styles)


