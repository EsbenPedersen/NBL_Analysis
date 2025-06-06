import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
import pandas as pd
from typing import Dict
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
    draft_sh = client.open('Season 24 Draft - test')
    dataframes = {}
    for sheet in draft_sh.worksheets():
        dataframes[f"draft_{sheet.title}"] = pd.DataFrame(sheet.get_all_values())

    # Fetch data from "Showcase Stats"
    stats_sh = client.open('Showcase Stats - test')
    for sheet in stats_sh.worksheets():
        dataframes[f"stats_{sheet.title}"] = pd.DataFrame(sheet.get_all_values())

    return dataframes 

if __name__ == "__main__":
    dataframes = get_google_sheets_data()
    print(dataframes)