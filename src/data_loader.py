import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
import pandas as pd
from typing import Dict, Optional, List
import logging
import os
import json
import re
import time

_DRAFT_CACHE: Dict[str, object] = {"data": None, "ts": 0.0}
_REGULAR_CACHE: Dict[str, object] = {"data": None, "ts": 0.0}
_DRAFT_TTL_SECONDS: int = int(os.environ.get("DRAFT_SHEETS_TTL_SECONDS", "5000"))
_REGULAR_TTL_SECONDS: int = int(os.environ.get("REGULAR_SHEETS_TTL_SECONDS", "5000"))


def get_google_sheets_data() -> Dict[str, pd.DataFrame]:
    """
    Fetches data from Google Sheets using service account credentials.

    It first tries to load credentials from an environment variable,
    which is suitable for production deployment. If not found, it falls back
    to a local 'credentials.json' file for local development.

    Returns:
        dict: A dictionary of pandas DataFrames, where each key is a sheet name.
    """
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # Serve from cache if fresh
    now = time.time()
    cached = _DRAFT_CACHE.get("data")
    if cached is not None and (now - float(_DRAFT_CACHE.get("ts", 0.0))) < _DRAFT_TTL_SECONDS:
        logging.info("Using cached draft sheets (age %.1fs)", now - float(_DRAFT_CACHE.get("ts", 0.0)))
        return cached  # type: ignore[return-value]
    
    creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    
    if creds_json_str:
        # Load credentials from environment variable (for production)
        creds_info = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    else:
        # Load credentials from local file (for local development)
        creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
        
    client = gspread.authorize(creds)

    try:
        # Fetch data from "Season 24 Draft"
        draft_sh = client.open('Season 24 Draft')
        dataframes: Dict[str, pd.DataFrame] = {}
        for sheet in draft_sh.worksheets():
            dataframes[f"draft_{sheet.title}"] = pd.DataFrame(sheet.get_all_values())

        # Fetch data from "Season 24 Draft - Snapshot"
        try:
            snapshot_sh = client.open('Season 24 Draft - Snapshot')
            for sheet in snapshot_sh.worksheets():
                dataframes[f"snapshot_{sheet.title}"] = pd.DataFrame(sheet.get_all_values())
        except gspread.SpreadsheetNotFound:
            logging.info("'Season 24 Draft - Snapshot' not found. Proceeding without it.")

        # Fetch data from "Showcase Stats", only the "Averages" tab
        try:
            stats_sh = client.open('Showcase Stats')
            try:
                worksheet = stats_sh.worksheet('Averages')
                dataframes['stats_Averages'] = pd.DataFrame(worksheet.get_all_values())
            except gspread.WorksheetNotFound:
                dataframes['stats_Averages'] = pd.DataFrame()
        except gspread.SpreadsheetNotFound:
            logging.info("'Showcase Stats' spreadsheet not found. Proceeding without it.")

        _DRAFT_CACHE["data"] = dataframes
        _DRAFT_CACHE["ts"] = now
        logging.info("Draft sheets fetched and cached (%d tabs)", len(dataframes))
        return dataframes
    except Exception as exc:
        # Fallback to cache on rate limiting or transient errors
        cached = _DRAFT_CACHE.get("data")
        if cached is not None:
            logging.warning("Draft fetch failed (%s). Serving cached data.", exc)
            return cached  # type: ignore[return-value]
        raise

if __name__ == "__main__":
    dataframes = get_google_sheets_data()
    print(dataframes)

def get_regular_season_data() -> Dict[str, pd.DataFrame]:
    """
    Fetches Regular Season data from Google Sheets by title.

    Prefers the static copy named 'Season 24 Table + Stats COPY' and falls back to
    'Season 24 Table + Stats' when the copy is unavailable.

    Returns a dictionary with cleaned DataFrames for:
    - 'standings': Team Standings
    - 'team_stats': Team Statistics
    - 'player_stats': Player Statistics
    """
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')

    if creds_json_str:
        creds_info = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    else:
        creds = Credentials.from_service_account_file('credentials.json', scopes=scope)

    # Serve from cache if fresh
    now = time.time()
    cached = _REGULAR_CACHE.get("data")
    if cached is not None and (now - float(_REGULAR_CACHE.get("ts", 0.0))) < _REGULAR_TTL_SECONDS:
        logging.info("Using cached regular season sheets (age %.1fs)", now - float(_REGULAR_CACHE.get("ts", 0.0)))
        return cached  # type: ignore[return-value]

    client = gspread.authorize(creds)

    # Open by title; prefer the COPY placeholder to avoid permission issues
    try:
        try:
            sh = client.open('Copy of Season 24 Table + Stats')
            logging.info("Using regular season sheet: 'Season 24 Table + Stats COPY'")
        except gspread.SpreadsheetNotFound:
            sh = client.open('Season 24 Table + Stats')
            logging.info("Using regular season sheet: 'Season 24 Table + Stats'")
    except gspread.SpreadsheetNotFound:
        # Provide clear guidance about access and a local CSV fallback option
        svc_email = getattr(creds, 'service_account_email', None)
        if svc_email:
            logging.error(
                "Regular season sheet not found or no access. Share the Google Sheet named 'Season 24 Table + Stats COPY' (or 'Season 24 Table + Stats') with service account '%s' (Viewer is sufficient).",
                svc_email,
            )
        else:
            logging.error("Regular season sheet not found or no access, and service account email unavailable from credentials.")

        csv_dir = os.environ.get('REGULAR_SEASON_CSV_DIR')
        if csv_dir and os.path.isdir(csv_dir):
            logging.info("Falling back to local CSV directory: %s", csv_dir)
            def _load_csv(name: str) -> pd.DataFrame:
                path = os.path.join(csv_dir, f"{name}.csv")
                if not os.path.isfile(path):
                    logging.warning("CSV not found: %s", path)
                    return pd.DataFrame()
                try:
                    df = pd.read_csv(path)
                    logging.info("Loaded CSV '%s' with shape %s", name, df.shape)
                    return df
                except Exception as exc:
                    logging.error("Failed reading CSV '%s': %s", path, exc)
                    return pd.DataFrame()

            return {
                'standings': _load_csv('Team Standings'),
                'team_stats': _load_csv('Team Statistics'),
                'player_stats': _load_csv('Player Statistics'),
            }

        # Return empty frames so app still runs; UI will show 'No data'
        return {'standings': pd.DataFrame(), 'team_stats': pd.DataFrame(), 'player_stats': pd.DataFrame()}

    # Log available worksheet titles to help diagnose naming mismatches
    try:
        available_titles: List[str] = [ws.title for ws in sh.worksheets()]
        logging.info("Regular season available tabs: %s", available_titles)
    except Exception as exc:
        logging.warning("Failed to list worksheet titles: %s", exc)
        available_titles = []

    def _normalize_title(value: str) -> str:
        if value is None:
            return ""
        # Lowercase, strip, collapse to alnum-only for robust matching
        simplified = re.sub(r"[^a-z0-9]+", "", value.lower())
        return simplified

    def _find_worksheet(desired_title: str) -> Optional[object]:
        # 1) Exact match first
        try:
            return sh.worksheet(desired_title)
        except gspread.WorksheetNotFound:
            pass
        # 2) Case-insensitive exact match
        desired_norm = desired_title.lower().strip()
        for ws in sh.worksheets():
            if ws.title.lower().strip() == desired_norm:
                logging.info("Matched worksheet by case-insensitive title: '%s' -> '%s'", desired_title, ws.title)
                return ws
        # 3) Normalized fuzzy containment match
        desired_key = _normalize_title(desired_title)
        for ws in sh.worksheets():
            ws_key = _normalize_title(ws.title)
            if desired_key and (desired_key in ws_key or ws_key in desired_key):
                logging.info("Matched worksheet by fuzzy title: '%s' ~ '%s'", desired_title, ws.title)
                return ws
        logging.warning("Worksheet not found for title '%s' (available: %s)", desired_title, available_titles)
        return None

    def _read_sheet(title: str) -> pd.DataFrame:
        ws = _find_worksheet(title)
        if ws is None:
            return pd.DataFrame()
        values = ws.get_all_values()
        if not values:
            logging.warning("Worksheet '%s' returned no values", ws.title)
            return pd.DataFrame()
        df = pd.DataFrame(values)
        if df.empty:
            logging.warning("Worksheet '%s' produced empty DataFrame after conversion", ws.title)
            return df
        # First row is header
        df.columns = df.iloc[0].astype(str)
        df = df.drop(index=0).reset_index(drop=True)
        # Trim headers
        df.columns = [c.strip() for c in df.columns]
        # Drop obviously empty/unnamed columns and deduplicate headers
        empty_or_unnamed = [c for c in df.columns if c == '' or c.lower().startswith('unnamed')]
        if empty_or_unnamed:
            df = df.drop(columns=empty_or_unnamed, errors='ignore')
        if df.columns.duplicated().any():
            logging.info("Dropping duplicated columns in '%s': %s", ws.title, list(df.columns[df.columns.duplicated()]))
            df = df.loc[:, ~df.columns.duplicated()]
        logging.info("Read '%s' with shape %s", ws.title, df.shape)
        return df

    standings_df = _read_sheet('Team Standings')
    team_stats_df = _read_sheet('Team Statistics')
    player_stats_df = _read_sheet('Player Statistics')

    # Clean numeric columns
    def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cleaned = df.copy()
        for col in list(cleaned.columns):
            # Skip obvious text columns
            if isinstance(col, tuple):
                base_name = str(col[0]).lower()
            else:
                base_name = str(col).lower()
            if base_name in {'abbreviation', 'team', 'general manager', 'name'}:
                continue
            # Series vs DataFrame guard (can happen with non-unique columns)
            col_values = cleaned[col]
            if isinstance(col_values, pd.DataFrame):
                for sub in col_values.columns:
                    series = col_values[sub].astype(str)
                    series = (series
                              .str.replace('%', '', regex=False)
                              .str.replace(',', '', regex=False)
                              .str.replace('#DIV/0!', '', regex=False)
                              .str.replace('—', '', regex=False)
                              .str.replace('-', '', regex=False)
                              .str.strip())
                    cleaned[(col, sub)] = pd.to_numeric(series, errors='coerce')
                continue
            series = col_values.astype(str)
            series = (series
                      .str.replace('%', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .str.replace('#DIV/0!', '', regex=False)
                      .str.replace('—', '', regex=False)
                      .str.replace('-', '', regex=False)
                      .str.strip())
            cleaned[col] = pd.to_numeric(series, errors='coerce')
        return cleaned

    standings_df = _coerce_numeric_cols(standings_df)
    # Drop the League Average summary row in Team Statistics (second line) if present
    def _drop_team_averages(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        temp = df.copy()
        team_col = 'Team' if 'Team' in temp.columns else None
        gm_col = 'General Manager' if 'General Manager' in temp.columns else None
        abbr_col = 'Abbreviation' if 'Abbreviation' in temp.columns else None
        if team_col or gm_col:
            mask = pd.Series(False, index=temp.index)
            if team_col:
                mask = mask | temp[team_col].astype(str).str.strip().str.lower().eq('league')
            if gm_col:
                mask = mask | temp[gm_col].astype(str).str.strip().str.lower().eq('average')
            if abbr_col:
                mask = mask | temp[abbr_col].astype(str).str.strip().eq('')
            dropped = int(mask.sum())
            if dropped:
                logging.info("Dropping %d 'League Average' summary rows from Team Statistics", dropped)
                temp = temp[~mask].reset_index(drop=True)
        return temp

    team_stats_df = _drop_team_averages(team_stats_df)
    team_stats_df = _coerce_numeric_cols(team_stats_df)

    from src.data_processing import clean_up_stats_df  # reuse for stats sheets
    player_stats_df = clean_up_stats_df(player_stats_df)

    # If Position is missing from player or all stats, try to map from Draft Snapshot sheets
    def _attach_positions(stats_df: pd.DataFrame) -> pd.DataFrame:
        if stats_df.empty or 'Name' not in stats_df.columns or 'Position' in stats_df.columns:
            return stats_df
        try:
            # Build mapping from Season 24 Draft - Snapshot (PG, SG, SF, PF, C)
            from src.data_processing import clean_up_available_players_df, _standardize_name  # type: ignore
            try:
                snapshot_sh = client.open('Season 24 Draft - Snapshot')
            except gspread.SpreadsheetNotFound:
                logging.info("'Season 24 Draft - Snapshot' not available for position mapping")
                return stats_df
            position_rows: List[pd.DataFrame] = []
            for pos_tab, pos in [('PG', 'PG'), ('SG', 'SG'), ('SF', 'SF'), ('PF', 'PF'), ('C', 'C')]:
                try:
                    ws = snapshot_sh.worksheet(pos_tab)
                    raw = pd.DataFrame(ws.get_all_values())
                    if raw.empty:
                        continue
                    cleaned = clean_up_available_players_df(raw.copy())
                    if cleaned.empty or 'Player' not in cleaned.columns:
                        continue
                    df_pos = cleaned[['Player']].rename(columns={'Player': 'Name'})
                    df_pos['Position'] = pos
                    position_rows.append(df_pos)
                except gspread.WorksheetNotFound:
                    continue
            if not position_rows:
                return stats_df
            positions_df = pd.concat(position_rows, ignore_index=True)
            # Standardized key merge to handle 'Last, First' vs 'First Last'
            stats_df = stats_df.copy()
            positions_df['name_key'] = positions_df['Name'].astype(str).apply(_standardize_name)
            stats_df['name_key'] = stats_df['Name'].astype(str).apply(_standardize_name)
            merged = pd.merge(stats_df, positions_df[['name_key', 'Position']], on='name_key', how='left')
            merged.drop(columns=['name_key'], inplace=True, errors='ignore')
            return merged
        except Exception as exc:
            logging.warning("Failed to map positions from snapshot: %s", exc)
            return stats_df

    player_stats_df = _attach_positions(player_stats_df)

    # Attach General Manager (and optionally Draft Team) to player_stats from Draft Sheet mapping
    def _attach_gm_from_draft(stats_df: pd.DataFrame) -> pd.DataFrame:
        if stats_df.empty or 'Name' not in stats_df.columns:
            return stats_df
        try:
            from src.data_processing import clean_up_draft_df, _standardize_name  # type: ignore
            draft_sh = client.open('Season 24 Draft')
            try:
                ws = draft_sh.worksheet('Draft Sheet')
            except gspread.WorksheetNotFound:
                # Fallback to first worksheet if named differently
                ws = draft_sh.worksheets()[0]
            raw = pd.DataFrame(ws.get_all_values())
            if raw.empty:
                return stats_df
            draft_df = clean_up_draft_df(raw.copy())
            team_headers = [c for c in draft_df.columns if c != 'Round']
            melted = draft_df.melt(id_vars='Round', var_name='DraftTeamHeader', value_name='CellText')
            melted = melted[melted['CellText'].astype(str).str.strip() != '']
            # Extract player name and GM from header
            def _extract_name(cell: str) -> str:
                import re as _re
                m = _re.match(r'^(.*?)\s*(?:\(\d+\))?$', str(cell).strip())
                return m.group(1).strip() if m else str(cell).strip()
            melted['Name'] = melted['CellText'].apply(_extract_name)
            # GM is header before '(' if present
            def _extract_gm(header: str) -> str:
                header = str(header)
                gm = header.split('(')[0].strip()
                return gm if gm else header
            melted['General Manager'] = melted['DraftTeamHeader'].apply(_extract_gm)
            # Build mapping on standardized name
            melted['name_key'] = melted['Name'].astype(str).apply(_standardize_name)
            gm_map = (
                melted.dropna(subset=['name_key'])
                .drop_duplicates(subset=['name_key'])[['name_key', 'General Manager', 'DraftTeamHeader']]
            )
            stats_df = stats_df.copy()
            stats_df['name_key'] = stats_df['Name'].astype(str).apply(_standardize_name)
            merged = pd.merge(stats_df, gm_map, on='name_key', how='left')
            merged.drop(columns=['name_key'], inplace=True, errors='ignore')
            # If Team missing, as a last resort fill with DraftTeamHeader (not ideal, but avoids empty UI)
            if 'Team' not in merged.columns and 'DraftTeamHeader' in merged.columns:
                merged.rename(columns={'DraftTeamHeader': 'Team'}, inplace=True)
            return merged
        except Exception as exc:
            logging.warning("Failed to map GM from Draft Sheet: %s", exc)
            return stats_df

    player_stats_df = _attach_gm_from_draft(player_stats_df)

    # Attach mappings from the centralized Season 24 Mapping spreadsheet
    def _attach_season24_mapping(stats_df: pd.DataFrame) -> pd.DataFrame:
        if stats_df.empty:
            return stats_df
        try:
            mapping_sh = client.open('Season 24 Mapping')
        except gspread.SpreadsheetNotFound:
            logging.info("'Season 24 Mapping' not found; skipping mapping enrichment")
            return stats_df

        def _read_ws(sh, title: str) -> pd.DataFrame:
            try:
                ws = sh.worksheet(title)
            except gspread.WorksheetNotFound:
                return pd.DataFrame()
            values = ws.get_all_values()
            if not values:
                return pd.DataFrame()
            dfm = pd.DataFrame(values)
            if dfm.empty:
                return dfm
            dfm.columns = dfm.iloc[0].astype(str)
            dfm = dfm.drop(index=0).reset_index(drop=True)
            dfm.columns = [c.strip() for c in dfm.columns]
            return dfm

        prp = _read_ws(mapping_sh, 'player_rating_position')
        ptm = _read_ws(mapping_sh, 'player_team_manager')

        # Normalize columns
        if not prp.empty:
            col_map = {}
            for c in prp.columns:
                lc = str(c).strip().lower()
                if lc in {'player', 'player name', 'name'}:
                    col_map[c] = 'Name'
                elif lc in {'rating', 'overall rating', 'overall'}:
                    col_map[c] = 'Overall Rating'
                elif lc in {'position', 'pos'}:
                    col_map[c] = 'Position'
            prp = prp.rename(columns=col_map)
            prp = prp[[c for c in ['Name', 'Overall Rating', 'Position'] if c in prp.columns]]
        if not ptm.empty:
            col_map = {}
            for c in ptm.columns:
                lc = str(c).strip().lower()
                if lc in {'player', 'player name', 'name'}:
                    col_map[c] = 'Name'
                elif lc in {'general manager', 'manager', 'gm'}:
                    col_map[c] = 'General Manager'
                elif lc in {'team', 'franchise'}:
                    col_map[c] = 'Team'
                elif lc in {'abbreviation', 'abbr', 'tm'}:
                    col_map[c] = 'Abbreviation'
            ptm = ptm.rename(columns=col_map)
            ptm = ptm[[c for c in ['Name', 'General Manager', 'Team', 'Abbreviation'] if c in ptm.columns]]

        if prp.empty and ptm.empty:
            return stats_df

        from src.data_processing import _standardize_name  # type: ignore
        base = stats_df.copy()
        base['name_key'] = base['Name'].astype(str).apply(_standardize_name) if 'Name' in base.columns else ''

        if not prp.empty:
            prp = prp.copy()
            prp['name_key'] = prp['Name'].astype(str).apply(_standardize_name)
            base = pd.merge(base, prp.drop(columns=['Name'], errors='ignore'), on='name_key', how='left', suffixes=('', '_map'))
            # Fill fields from mapping
            if 'Overall Rating_map' in base.columns:
                base['Overall Rating'] = pd.to_numeric(base.get('Overall Rating', pd.Series(index=base.index)), errors='coerce')
                base['Overall Rating'] = base['Overall Rating'].fillna(pd.to_numeric(base['Overall Rating_map'], errors='coerce'))
                base.drop(columns=['Overall Rating_map'], inplace=True, errors='ignore')
            if 'Position_map' in base.columns:
                base['Position'] = base.get('Position').fillna(base['Position_map']) if 'Position' in base.columns else base['Position_map']
                base.drop(columns=['Position_map'], inplace=True, errors='ignore')

        if not ptm.empty:
            ptm = ptm.copy()
            ptm['name_key'] = ptm['Name'].astype(str).apply(_standardize_name)
            base = pd.merge(base, ptm.drop(columns=['Name'], errors='ignore'), on='name_key', how='left', suffixes=('', '_map2'))
            for tgt in ['Team', 'General Manager', 'Abbreviation']:
                map_col = f'{tgt}_map2'
                if map_col in base.columns:
                    base[tgt] = base.get(tgt).fillna(base[map_col]) if tgt in base.columns else base[map_col]
                    base.drop(columns=[map_col], inplace=True, errors='ignore')

        # Add missing players from mapping (so roster shows players with no stats yet)
        mapping_union = pd.DataFrame()
        union_sources = []
        if not prp.empty:
            union_sources.append(prp[['Name']].copy())
        if not ptm.empty:
            union_sources.append(ptm[['Name']].copy())
        if union_sources:
            mapping_union = pd.concat(union_sources, ignore_index=True).drop_duplicates()
        if not mapping_union.empty:
            mapping_union['name_key'] = mapping_union['Name'].astype(str).apply(_standardize_name)
            present_keys = set(base['name_key'])
            missing = mapping_union[~mapping_union['name_key'].isin(present_keys)]
            if not missing.empty:
                # Build rows from mapping info
                add_rows = missing.copy()
                # Attach attributes from mappings if available
                if not prp.empty:
                    add_rows = pd.merge(add_rows, prp[['name_key', 'Overall Rating', 'Position']], on='name_key', how='left')
                if not ptm.empty:
                    add_rows = pd.merge(add_rows, ptm[['name_key', 'Team', 'General Manager', 'Abbreviation']], on='name_key', how='left')
                # Ensure required columns exist
                for col in ['Position', 'Team', 'General Manager', 'Abbreviation', 'Overall Rating']:
                    if col not in add_rows.columns:
                        add_rows[col] = pd.NA
                # Align to base columns
                keep_cols = [c for c in base.columns if c != 'name_key']
                # Build new frame with base columns
                new_frame = pd.DataFrame(columns=base.columns)
                for _, r in add_rows.iterrows():
                    row = {k: pd.NA for k in base.columns}
                    row['name_key'] = r['name_key']
                    row['Name'] = r.get('Name', pd.NA)
                    row['Position'] = r.get('Position', pd.NA)
                    row['Team'] = r.get('Team', pd.NA)
                    row['General Manager'] = r.get('General Manager', pd.NA)
                    row['Abbreviation'] = r.get('Abbreviation', pd.NA)
                    row['Overall Rating'] = r.get('Overall Rating', pd.NA)
                    new_frame.loc[len(new_frame)] = row
                base = pd.concat([base, new_frame], ignore_index=True)

        base.drop(columns=['name_key'], inplace=True, errors='ignore')
        return base

    player_stats_df = _attach_season24_mapping(player_stats_df)

    # Optionally attach centralized roster mapping (Player-Position-Team-Manager)
    def _attach_roster_mapping(stats_df: pd.DataFrame) -> pd.DataFrame:
        if stats_df.empty or 'Name' not in stats_df.columns:
            return stats_df
        try:
            mapping_title = os.environ.get('ROSTER_MAPPING_SHEET_NAME', 'Roster Mapping')
            try:
                roster_ws = _find_worksheet(mapping_title)
                # If not found by title in the same workbook, try open by name
                if roster_ws is None:
                    roster_sh = client.open(mapping_title)  # may raise
                    # prefer first worksheet
                    roster_ws = roster_sh.worksheets()[0]
            except Exception:
                return stats_df
            values = roster_ws.get_all_values()
            if not values:
                return stats_df
            rm = pd.DataFrame(values)
            if rm.empty:
                return stats_df
            # First row header
            rm.columns = rm.iloc[0].astype(str)
            rm = rm.drop(index=0).reset_index(drop=True)
            # Normalize expected columns
            col_map = {}
            for c in rm.columns:
                lc = str(c).strip().lower()
                if lc in {'player', 'name'}:
                    col_map[c] = 'Name'
                elif lc in {'position', 'pos'}:
                    col_map[c] = 'Position'
                elif lc in {'team', 'franchise'}:
                    col_map[c] = 'Team'
                elif lc in {'general manager', 'manager', 'gm'}:
                    col_map[c] = 'General Manager'
            if col_map:
                rm = rm.rename(columns=col_map)
            required = [c for c in ['Name', 'Position', 'Team', 'General Manager'] if c in rm.columns]
            if not required:
                return stats_df
            # Standardized key merge
            from src.data_processing import _standardize_name  # type: ignore
            rm = rm[[c for c in ['Name', 'Position', 'Team', 'General Manager'] if c in rm.columns]].copy()
            rm['name_key'] = rm['Name'].astype(str).apply(_standardize_name)
            enriched = stats_df.copy()
            enriched['name_key'] = enriched['Name'].astype(str).apply(_standardize_name)
            merged = pd.merge(enriched, rm.drop(columns=['Name'], errors='ignore'), on='name_key', how='left', suffixes=('', '_map'))
            merged.drop(columns=['name_key'], inplace=True, errors='ignore')
            # Fill missing core fields from mapping
            for tgt_col in ['Position', 'Team', 'General Manager']:
                map_col = f"{tgt_col}_map"
                if map_col in merged.columns:
                    merged[tgt_col] = merged[tgt_col].fillna(merged[map_col]) if tgt_col in merged.columns else merged[map_col]
                    merged.drop(columns=[map_col], inplace=True, errors='ignore')
            return merged
        except Exception as exc:
            logging.warning("Failed to attach roster mapping: %s", exc)
            return stats_df

    player_stats_df = _attach_roster_mapping(player_stats_df)

    result = {
        'standings': standings_df,
        'team_stats': team_stats_df,
        'player_stats': player_stats_df,
    }
    # Final shapes log
    for key, frame in result.items():
        try:
            logging.info("regular.%s final shape: %s", key, getattr(frame, 'shape', None))
        except Exception:
            pass
    _REGULAR_CACHE["data"] = result
    _REGULAR_CACHE["ts"] = now
    return result