import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
import pandas as pd
from typing import Dict
import logging
import os
import json

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
    
    creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    
    if creds_json_str:
        # Load credentials from environment variable (for production)
        creds_info = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    else:
        # Load credentials from local file (for local development)
        creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
        
    client = gspread.authorize(creds)

    # Fetch data from "Season 24 Draft"
    draft_sh = client.open('Season 24 Draft')
    dataframes = {}
    for sheet in draft_sh.worksheets():
        dataframes[f"draft_{sheet.title}"] = pd.DataFrame(sheet.get_all_values())

    # Fetch data from "Season 24 Draft - Snapshot"
    try:
        snapshot_sh = client.open('Season 24 Draft - Snapshot')
        for sheet in snapshot_sh.worksheets():
            # Storing with a 'snapshot_' prefix to distinguish them
            dataframes[f"snapshot_{sheet.title}"] = pd.DataFrame(sheet.get_all_values())
    except gspread.SpreadsheetNotFound:
        print("Warning: 'Season 24 Draft - Snapshot' not found. Proceeding without it.")

    # Fetch data from "Showcase Stats", only the "Averages" tab
    stats_sh = client.open('Showcase Stats')
    try:
        worksheet = stats_sh.worksheet('Averages')
        dataframes['stats_Averages'] = pd.DataFrame(worksheet.get_all_values())
    except gspread.WorksheetNotFound:
        # If "Averages" sheet is not found, create an empty DataFrame
        dataframes['stats_Averages'] = pd.DataFrame()

    return dataframes 

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

    client = gspread.authorize(creds)

    # Open by title; prefer the COPY placeholder to avoid permission issues
    try:
        try:
            sh = client.open('Season 24 Table + Stats COPY')
            logging.info("Using regular season sheet: 'Season 24 Table + Stats COPY'")
        except gspread.SpreadsheetNotFound:
            sh = client.open('Season 24 Table + Stats')
            logging.info("Using regular season sheet: 'Season 24 Table + Stats'")
    except gspread.SpreadsheetNotFound:
        # Return empty frames so app still runs; UI will show 'No data'
        return {'standings': pd.DataFrame(), 'team_stats': pd.DataFrame(), 'player_stats': pd.DataFrame()}

    def _read_sheet(title: str) -> pd.DataFrame:
        try:
            ws = sh.worksheet(title)
        except gspread.WorksheetNotFound:
            return pd.DataFrame()
        df = pd.DataFrame(ws.get_all_values())
        if df.empty:
            return df
        # First row is header
        df.columns = df.iloc[0].astype(str)
        df = df.drop(index=0).reset_index(drop=True)
        # Trim headers
        df.columns = [c.strip() for c in df.columns]
        return df

    standings_df = _read_sheet('Team Standings')
    team_stats_df = _read_sheet('Team Statistics')
    player_stats_df = _read_sheet('Player Statistics')

    # Clean numeric columns
    def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cleaned = df.copy()
        for col in cleaned.columns:
            # Skip obvious text columns
            if col.lower() in {'abbreviation', 'team', 'general manager', 'name'}:
                continue
            # Remove % signs and commas and coerce
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('#DIV/0!', '', regex=False)
                .str.strip()
            )
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
        return cleaned

    standings_df = _coerce_numeric_cols(standings_df)
    team_stats_df = _coerce_numeric_cols(team_stats_df)

    from src.data_processing import clean_up_stats_df  # reuse for player stats
    player_stats_df = clean_up_stats_df(player_stats_df)

    return {
        'standings': standings_df,
        'team_stats': team_stats_df,
        'player_stats': player_stats_df,
    }