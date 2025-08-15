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
                dcc.Graph(id='rs-team-graph', style={'height': '50vh', 'width': '100%'})
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
                dcc.Graph(id='rs-player-graph', style={'height': '50vh', 'width': '100%'})
            ])
        ]), md=6),
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader(html.H3("Trade Targets")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='rs-team-select', placeholder='Select Team'), md=6),
                dbc.Col(dbc.Input(id='rs-trade-topn', type='number', value=3, min=1, max=10, step=1, placeholder='Top N per Position'), md=3),
            ], className='mb-3 g-2'),
            html.Div(id='rs-trade-summary', className='mb-3'),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H5("Current Roster")),
                    dbc.CardBody(html.Div(id='rs-trade-underperformers'))
                ]), md=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H5("Ideal Trade Targets")),
                    dbc.CardBody(html.Div(id='rs-trade-ideal'))
                ]), md=6),
            ], className='mb-3'),
            dbc.Card([
                dbc.CardHeader(html.H5("Budget Trade Targets")),
                dbc.CardBody(html.Div(id='rs-trade-budget'))
            ])
        ])
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

    # Round win Percentage to 2 decimals if present
    if 'Percentage' in standings_df.columns:
        try:
            standings_df = standings_df.copy()
            standings_df['Percentage'] = pd.to_numeric(standings_df['Percentage'], errors='coerce').round(2)
        except Exception:
            pass

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
    Output('rs-team-select', 'options'),
    Input('rs-data-store', 'data')
)
def rs_team_select_options(json_payload: Optional[Dict[str, str]]) -> List[Dict[str, str]]:
    if not json_payload:
        return []
    standings = pd.read_json(StringIO(json_payload['standings']), orient='split') if 'standings' in json_payload else pd.DataFrame()
    if standings.empty:
        return []
    label_col = 'Team' if 'Team' in standings.columns else standings.columns[0]
    return [{'label': t, 'value': t} for t in sorted(standings[label_col].dropna().unique())]


@dash.callback(
    Output('rs-team-select', 'value'),
    Input('rs-team-select', 'options')
)
def rs_team_select_default(options: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not options:
        return None
    values = {opt.get('value') for opt in options}
    return 'Net Profits' if 'Net Profits' in values else None


@dash.callback(
    [Output('rs-trade-summary', 'children'),
     Output('rs-trade-underperformers', 'children'),
     Output('rs-trade-ideal', 'children'),
     Output('rs-trade-budget', 'children')],
    [Input('rs-team-select', 'value'), Input('rs-trade-topn', 'value'), Input('rs-data-store', 'data')]
)
def rs_trade_suggestions(team_name: Optional[str], topn: Optional[int], json_payload: Optional[Dict[str, str]]) -> Tuple[Any, Any, Any, Any]:
    if not json_payload or not team_name:
        return "Select a team to see trade targets", "", "", ""
    team_stats = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if 'team_stats' in json_payload else pd.DataFrame()
    player_stats = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if 'player_stats' in json_payload else pd.DataFrame()
    if team_stats.empty or player_stats.empty:
        return "No data", "", "", ""

    # Identify team abbreviation if available
    # Determine the team key for player_stats robustly
    if 'Team' in player_stats.columns:
        team_key: Optional[str] = 'Team'
    elif 'Abbreviation' in player_stats.columns:
        team_key = 'Abbreviation'
    else:
        candidates_keys = [c for c in player_stats.columns if c.lower() in {'team', 'abbreviation', 'abbr', 'tm'}]
        team_key = candidates_keys[0] if candidates_keys else None
    if not team_key:
        return "Player stats missing team identifier column", "", "", ""
    opp_key = team_key
    abbr_col = 'Abbreviation' if 'Abbreviation' in team_stats.columns else None
    selected = team_stats[team_stats[team_key] == team_name] if team_key in team_stats.columns else pd.DataFrame()
    if selected.empty and abbr_col:
        selected = team_stats[team_stats[abbr_col] == team_name]
    if selected.empty:
        return "Team not found", "", "", ""

    pos_col = 'Position' if 'Position' in player_stats.columns else None
    if not pos_col:
        return "Position data unavailable; cannot compute trade targets", "", "", ""
    topn = int(topn or 3)

    # Players on selected team (robust across team/gm/abbr with case-insensitive match)
    p_team_col = 'Team' if 'Team' in player_stats.columns else None
    p_abbr_col = 'Abbreviation' if 'Abbreviation' in player_stats.columns else None
    p_gm_col = 'General Manager' if 'General Manager' in player_stats.columns else None
    p_header_col = 'DraftTeamHeader' if 'DraftTeamHeader' in player_stats.columns else None
    sel_abbr_val = selected.iloc[0][abbr_col] if (abbr_col and abbr_col in selected.columns) else None
    sel_team_val = selected.iloc[0]['Team'] if 'Team' in selected.columns else team_name
    sel_gm_val = selected.iloc[0]['General Manager'] if 'General Manager' in selected.columns else None

    def _ci_eq(series: pd.Series, wanted: Optional[str]) -> pd.Series:
        if not wanted or series is None:
            return pd.Series(False, index=player_stats.index)
        return series.astype(str).str.strip().str.lower() == str(wanted).strip().lower()

    masks: List[pd.Series] = []
    if p_team_col:
        masks.append(_ci_eq(player_stats[p_team_col], team_name))
        masks.append(_ci_eq(player_stats[p_team_col], sel_team_val))
        if sel_abbr_val is not None:
            masks.append(_ci_eq(player_stats[p_team_col], sel_abbr_val))
    if p_abbr_col:
        masks.append(_ci_eq(player_stats[p_abbr_col], team_name))
        if sel_abbr_val is not None:
            masks.append(_ci_eq(player_stats[p_abbr_col], sel_abbr_val))
    if p_gm_col and sel_gm_val:
        masks.append(_ci_eq(player_stats[p_gm_col], sel_gm_val))
    if p_header_col and sel_gm_val:
        # Header often begins with GM name
        series = player_stats[p_header_col].astype(str).str.strip().str.lower()
        masks.append(series.str.startswith(str(sel_gm_val).strip().lower()))

    players_on_team = pd.DataFrame()
    if masks:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        players_on_team = player_stats[combined].copy()
    if not players_on_team.empty:
        players_on_team = players_on_team.drop_duplicates()

    # Helper: find columns by fuzzy name
    def _find_col(df: pd.DataFrame, needles: List[str]) -> Optional[str]:
        for col in df.columns:
            lc = str(col).lower()
            for n in needles:
                if n in lc:
                    return col
        return None

    # Determine team weaknesses using Off/Def Ratings if available; otherwise try proxies
    off_col = _find_col(team_stats, ['offensive rating', 'off rating', 'offrtg', 'off rtg', 'off efficiency', 'off eff'])
    def_col = _find_col(team_stats, ['defensive rating', 'def rating', 'defrtg', 'def rtg', 'def efficiency', 'def eff'])
    net_col = _find_col(team_stats, ['net rating', 'net efficiency', 'net eff'])
    ppp_col = _find_col(team_stats, ['points per possession', 'pts per possession', 'ppp'])
    opp_ppp_col = _find_col(team_stats, ['opponent points per possession', 'opp points per possession', 'opp ppp'])
    ppg_col = _find_col(team_stats, ['points per game', 'pts per game', 'ppg'])
    opp_ppg_col = _find_col(team_stats, ['opponent points per game', 'opp points per game', 'opp ppg'])

    needs: List[str] = []
    summary_bits: List[str] = []
    try:
        row = selected.iloc[0]
        if off_col and off_col in team_stats.columns:
            off_mean = pd.to_numeric(team_stats[off_col], errors='coerce').mean()
            off_std = pd.to_numeric(team_stats[off_col], errors='coerce').std()
            off_val = pd.to_numeric(pd.Series([row[off_col]]), errors='coerce').iloc[0]
            if pd.notna(off_val) and pd.notna(off_mean):
                if off_val < off_mean - (0.35 * (off_std if pd.notna(off_std) else 0)):
                    needs.append('offense')
                    summary_bits.append(f"Offense below league average ({off_col}: {off_val:.2f})")
        if def_col and def_col in team_stats.columns:
            def_mean = pd.to_numeric(team_stats[def_col], errors='coerce').mean()
            def_std = pd.to_numeric(team_stats[def_col], errors='coerce').std()
            def_val = pd.to_numeric(pd.Series([row[def_col]]), errors='coerce').iloc[0]
            if pd.notna(def_val) and pd.notna(def_mean):
                if def_val > def_mean + (0.35 * (def_std if pd.notna(def_std) else 0)):
                    needs.append('defense')
                    summary_bits.append(f"Defense below league average ({def_col}: {def_val:.2f})")
        if not needs and net_col and net_col in team_stats.columns:
            net_mean = pd.to_numeric(team_stats[net_col], errors='coerce').mean()
            net_std = pd.to_numeric(team_stats[net_col], errors='coerce').std()
            net_val = pd.to_numeric(pd.Series([row[net_col]]), errors='coerce').iloc[0]
            if pd.notna(net_val) and pd.notna(net_mean) and net_val < net_mean - (0.35 * (net_std if pd.notna(net_std) else 0)):
                needs = ['offense', 'defense']
                summary_bits.append(f"Overall efficiency below average ({net_col}: {net_val:.2f})")
    except Exception:
        pass
    if not needs:
        # Try PPP/PPG proxies
        try:
            row = selected.iloc[0]
            if ppp_col and opp_ppp_col and ppp_col in team_stats.columns and opp_ppp_col in team_stats.columns:
                ppp_mean = pd.to_numeric(team_stats[ppp_col], errors='coerce').mean()
                opp_ppp_mean = pd.to_numeric(team_stats[opp_ppp_col], errors='coerce').mean()
                ppp_val = pd.to_numeric(pd.Series([row[ppp_col]]), errors='coerce').iloc[0]
                opp_ppp_val = pd.to_numeric(pd.Series([row[opp_ppp_col]]), errors='coerce').iloc[0]
                if pd.notna(ppp_val) and pd.notna(ppp_mean) and ppp_val < ppp_mean:
                    needs.append('offense')
                    summary_bits.append(f"Offense below league average ({ppp_col}: {ppp_val:.2f})")
                if pd.notna(opp_ppp_val) and pd.notna(opp_ppp_mean) and opp_ppp_val > opp_ppp_mean:
                    needs.append('defense')
                    summary_bits.append(f"Defense below league average ({opp_ppp_col}: {opp_ppp_val:.2f})")
        except Exception:
            pass
    if not needs:
        try:
            row = selected.iloc[0]
            if ppg_col and opp_ppg_col and ppg_col in team_stats.columns and opp_ppg_col in team_stats.columns:
                ppg_mean = pd.to_numeric(team_stats[ppg_col], errors='coerce').mean()
                opp_ppg_mean = pd.to_numeric(team_stats[opp_ppg_col], errors='coerce').mean()
                ppg_val = pd.to_numeric(pd.Series([row[ppg_col]]), errors='coerce').iloc[0]
                opp_ppg_val = pd.to_numeric(pd.Series([row[opp_ppg_col]]), errors='coerce').iloc[0]
                if pd.notna(ppg_val) and pd.notna(ppg_mean) and ppg_val < ppg_mean:
                    needs.append('offense')
                    summary_bits.append(f"Offense below league average ({ppg_col}: {ppg_val:.2f})")
                if pd.notna(opp_ppg_val) and pd.notna(opp_ppg_mean) and opp_ppg_val > opp_ppg_mean:
                    needs.append('defense')
                    summary_bits.append(f"Defense below league average ({opp_ppg_col}: {opp_ppg_val:.2f})")
        except Exception:
            pass
    if not needs:
        needs = ['offense', 'defense']
        summary_bits.append("Metrics missing for this team; assuming balanced needs")

    # Build value metric with fallbacks and ensure it has variance and non-null data
    preferred_metrics = ['VORP', 'Plus Minus', 'Game Score', 'Overall Rating', 'PER', 'TS%', 'EFG%']
    value_metric: Optional[str] = None
    for m in preferred_metrics:
        if m in player_stats.columns:
            series = pd.to_numeric(player_stats[m], errors='coerce')
            if series.notna().sum() >= 3 and series.std(skipna=True) > 0:
                value_metric = m
                break
    if value_metric is None:
        nums_df = player_stats.select_dtypes(include='number')
        for col in nums_df.columns:
            series = pd.to_numeric(nums_df[col], errors='coerce')
            if series.notna().sum() >= 3 and series.std(skipna=True) > 0:
                value_metric = col
                break

    # Compute team position strength (median of value_metric)
    team_pos_strength = pd.Series(dtype=float)
    if value_metric and not players_on_team.empty:
        try:
            team_pos_strength = players_on_team.groupby(pos_col)[value_metric].median().rename('team_pos_strength')
        except Exception:
            team_pos_strength = pd.Series(dtype=float)

    # Prepare candidate pool: players not on selected team
    candidates = player_stats.copy()
    if p_team_col and p_team_col in candidates.columns:
        candidates = candidates[candidates[p_team_col] != team_name]
        if sel_abbr_val is not None:
            candidates = candidates[candidates[p_team_col] != sel_abbr_val]
    if p_abbr_col and p_abbr_col in candidates.columns:
        candidates = candidates[candidates[p_abbr_col] != team_name]
        if sel_abbr_val is not None:
            candidates = candidates[candidates[p_abbr_col] != sel_abbr_val]

    # Ensure Stocks exists if possible
    if 'Stocks' not in candidates.columns and {'Steals', 'Blocks'}.issubset(set(candidates.columns)):
        try:
            candidates['Stocks'] = pd.to_numeric(candidates['Steals'], errors='coerce').fillna(0) + pd.to_numeric(candidates['Blocks'], errors='coerce').fillna(0)
        except Exception:
            pass

    # Fit scoring using z-scores
    def _z(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors='coerce')
        mean = s_num.mean()
        std = s_num.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=s.index)
        return (s_num - mean) / std

    offense_cols = [c for c in ['Points', 'Assists', 'FG Attempted', 'FG Attempt', 'EFG%', 'TS%'] if c in candidates.columns]
    if 'FG Attempted' not in offense_cols and 'FG Attempt' in offense_cols:
        # unify naming
        candidates.rename(columns={'FG Attempt': 'FG Attempted'}, inplace=True)
        offense_cols = [c if c != 'FG Attempt' else 'FG Attempted' for c in offense_cols]
    defense_cols = [c for c in ['Rebounds', 'Blocks', 'Steals', 'Stocks'] if c in candidates.columns]

    if not offense_cols and not defense_cols:
        return "Insufficient player metrics to compute fit", "", "", ""

    # Compose fit score
    off_fit = pd.Series(0, index=candidates.index)
    if offense_cols:
        off_fit = sum((_z(candidates[c]) for c in offense_cols)) / float(len(offense_cols))
    def_fit = pd.Series(0, index=candidates.index)
    if defense_cols:
        def_fit = sum((_z(candidates[c]) for c in defense_cols)) / float(len(defense_cols))

    if set(needs) == {'offense'}:
        candidates['FitScore'] = off_fit
    elif set(needs) == {'defense'}:
        candidates['FitScore'] = def_fit
    else:
        candidates['FitScore'] = 0.5 * off_fit + 0.5 * def_fit

    # Ideal: top by FitScore per position
    ideal_rows: List[pd.DataFrame] = []
    for pos, grp in candidates.groupby(pos_col):
        pick = grp.sort_values('FitScore', ascending=False).head(topn)
        ideal_rows.append(pick)
    ideal_df = pd.concat(ideal_rows, ignore_index=True) if ideal_rows else pd.DataFrame()

    # Budget: below 60th percentile by value_metric, but high FitScore
    budget_df = pd.DataFrame()
    if value_metric and value_metric in candidates.columns and not candidates.empty:
        vm = pd.to_numeric(candidates[value_metric], errors='coerce')
        threshold = vm.quantile(0.60)
        budget_pool = candidates[vm <= threshold]
        b_rows: List[pd.DataFrame] = []
        for pos, grp in budget_pool.groupby(pos_col):
            pick = grp.sort_values('FitScore', ascending=False).head(topn)
            b_rows.append(pick)
        budget_df = pd.concat(b_rows, ignore_index=True) if b_rows else pd.DataFrame()

    # Current roster table (include players with zero minutes or missing stats)
    roster_df = pd.DataFrame()
    if not players_on_team.empty:
        roster_df = players_on_team.copy()
        # Prefer sorting by Minutes if available
        min_col = next((c for c in roster_df.columns if str(c).lower() in {'minutes', 'mins'}), None)
        if min_col:
            roster_df[min_col] = pd.to_numeric(roster_df[min_col], errors='coerce')
            roster_df = roster_df.sort_values(min_col, ascending=False)

    def _make_table(
        df: pd.DataFrame,
        cols_priority: List[str]
    ) -> Any:
        if df is None or df.empty:
            return "No data"

        show_cols: List[str] = [c for c in cols_priority if c in df.columns]
        default_cols = ['Name', 'Team', pos_col, 'FitScore', 'VORP', 'Overall Rating',
                        'Points', 'Assists', 'Rebounds', 'Stocks', 'EFG%', 'TS%']
        for c in default_cols:
            if c and c in df.columns and c not in show_cols:
                show_cols.append(c)

        if not show_cols:
            show_cols = list(df.columns)

        df_display = df.loc[:, show_cols].copy()
        num_cols = df_display.select_dtypes(include="number").columns
        df_display[num_cols] = df_display[num_cols].apply(pd.to_numeric, errors='coerce').round(3)

        styles: List[Dict[str, Any]] = []
        for hcol in ['FitScore', 'VORP', 'Overall Rating']:
            if hcol in df_display.columns:
                styles.extend(generate_heatmap_style(df_display, hcol))

        return dash_table.DataTable(
            data=df_display.to_dict('records'),
            columns=[{"name": c, "id": c} for c in df_display.columns],
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'textAlign': 'center',
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'border': '1px solid rgb(80, 80, 80)'
            },
            style_data_conditional=styles,
            page_size=20,
        )

    def _make_roster_table(df: pd.DataFrame) -> Any:
        if df.empty:
            return "No data"
        # Normalize display names
        try:
            from src.data_processing import normalize_person_name  # type: ignore
            if 'Name' in df.columns:
                df = df.copy()
                df['Name'] = df['Name'].astype(str).apply(normalize_person_name)
            if 'General Manager' in df.columns:
                df['General Manager'] = df['General Manager'].astype(str).apply(normalize_person_name)
        except Exception:
            pass
        desired_cols = ['Name', pos_col or 'Position', 'Team', 'General Manager', 'Minutes', 'Points', 'Assists', 'Rebounds', 'Blocks', 'Steals', 'Stocks', 'EFG%', 'TS%', 'Plus Minus', 'VORP', 'Overall Rating']
        show_cols = [c for c in desired_cols if c in df.columns]
        if not show_cols:
            show_cols = list(df.columns)
        disp = df[show_cols].copy()
        # Round numerics; then replace NaN with '-'
        for c in disp.columns:
            if pd.api.types.is_numeric_dtype(disp[c]):
                disp[c] = pd.to_numeric(disp[c], errors='coerce').round(3)
        disp = disp.where(pd.notna(disp), '-')
        return dash_table.DataTable(
            data=disp.to_dict('records'),
            columns=[{"name": c, "id": c} for c in disp.columns],
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'border': '1px solid rgb(80, 80, 80)'},
            page_size=20,
        )

    # Build summary text
    summary_text = "; ".join(summary_bits) if summary_bits else "Balanced needs"
    summary = html.Div([
        html.Strong("Team Needs: "), html.Span(summary_text)
    ])

    return (
        summary,
        _make_roster_table(roster_df),
        _make_table(ideal_df, ['Name', pos_col or 'Position', 'Team', 'FitScore', value_metric or '']),
        _make_table(budget_df, ['Name', pos_col or 'Position', 'Team', 'FitScore', value_metric or ''])
    )


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
    hover_data = [c for c in df.columns if c not in {x, y}]
    gm_series = df['General Manager'] if 'General Manager' in df.columns else None
    try:
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None,
                         size=size_series, text=gm_series, hover_data=hover_data)
    except Exception:
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, size=size_series, hover_data=hover_data)
    fig.update_traces(textposition='top center', textfont_size=10)
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
    hover_data = [c for c in df.columns if c not in {x, y}]
    try:
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None,
                         size=size_series, text=df['Name'] if 'Name' in df.columns else None,
                         hover_data=hover_data)
    except Exception:
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, size=size_series, hover_data=hover_data)
    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_layout(hovermode='closest')
    return fig


@dash.callback(
    Output('rs-team-graph', 'figure', allow_duplicate=True),
    [Input('rs-team-text-toggle', 'value'), Input('rs-data-store', 'data'), Input('rs-team-x', 'value'), Input('rs-team-y', 'value'), Input('rs-team-color', 'value'), Input('rs-team-size', 'value')],
    prevent_initial_call=True
)
def rs_team_text_labels(toggle, json_payload, x, y, color, size):
    # Rebuild from base df to guarantee consistent text-length with data
    df = pd.read_json(StringIO(json_payload['team_stats']), orient='split') if json_payload and 'team_stats' in json_payload else pd.DataFrame()
    fig = rs_team_plot(json_payload, x, y, color, size)
    show = bool(toggle and 'on' in toggle)
    if show and not df.empty and 'General Manager' in df.columns:
        fig.update_traces(text=df['General Manager'])
    else:
        fig.update_traces(text=None)
    return fig


@dash.callback(
    Output('rs-player-graph', 'figure', allow_duplicate=True),
    [Input('rs-player-text-toggle', 'value'), Input('rs-data-store', 'data'), Input('rs-player-x', 'value'), Input('rs-player-y', 'value'), Input('rs-player-color', 'value'), Input('rs-player-size', 'value')],
    prevent_initial_call=True
)
def rs_player_text_labels(toggle, json_payload, x, y, color, size):
    df = pd.read_json(StringIO(json_payload['player_stats']), orient='split') if json_payload and 'player_stats' in json_payload else pd.DataFrame()
    fig = rs_player_plot(json_payload, x, y, color, size)
    show = bool(toggle and 'on' in toggle)
    if show and not df.empty and 'Name' in df.columns:
        fig.update_traces(text=df['Name'])
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
    # Ensure Team column exists by mirroring from Abbreviation when that is the only identifier
    if 'Team' not in df.columns and 'Abbreviation' in df.columns:
        df = df.rename(columns={'Abbreviation': 'Team'})
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
        show_cols = [c for c in ['Name', 'VORP', 'Plus Minus'] if c in top_overall.columns]
        return dash_table.DataTable(
            data=top_overall[show_cols].to_dict('records'),
            columns=[{"name": c, "id": c} for c in show_cols],
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'fontSize': '12px', 'padding': '4px 6px'},
            style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'border': '1px solid rgb(80, 80, 80)', 'fontSize': '12px', 'padding': '4px 6px'}
        )

    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    cols: List[Any] = []
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
        # Columns to display: Name, VORP, Plus Minus (subset to available)
        show_cols = [c for c in ['Name', 'VORP', 'Plus Minus'] if c in top_pos.columns]
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
            html.H5(pos, className="mb-2 text-center"),
            dash_table.DataTable(
                data=top_pos[show_cols].to_dict('records'),
                columns=[{"name": c, "id": c} for c in show_cols],
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'fontSize': '12px', 'padding': '4px 6px'},
                style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'border': '1px solid rgb(80, 80, 80)', 'fontSize': '12px', 'padding': '4px 6px'},
                style_data_conditional=styles,
                style_table={'width': '100%'},
            )
        ])
        cols.append(dbc.Col(table, xs=12, sm=6, md=4, lg=2, className="d-flex flex-column", style={'flex': '1 1 18%'}))

    if not cols:
        return "No data"
    return dbc.Row(cols, className="g-2 justify-content-between")


