import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from typing import Dict

def get_google_sheets_data() -> Dict[str, pd.DataFrame]:
    """
    Authenticates with Google Sheets API using service account credentials
    and fetches all worksheets from "Season 24 Draft" and "Showcase Stats"
    spreadsheets.

    Returns:
        A dictionary mapping sheet titles to pandas DataFrames.
    """
    # Define the scope
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]

    # Add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)

    # Authorize the clientsheet
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