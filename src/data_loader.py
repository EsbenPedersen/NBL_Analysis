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
_DRAFT_TTL_SECONDS: int = int(os.environ.get("DRAFT_SHEETS_TTL_SECONDS", "300"))
_REGULAR_TTL_SECONDS: int = int(os.environ.get("REGULAR_SHEETS_TTL_SECONDS", "300"))


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
    team_stats_df = _coerce_numeric_cols(team_stats_df)

    from src.data_processing import clean_up_stats_df  # reuse for player stats
    player_stats_df = clean_up_stats_df(player_stats_df)

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