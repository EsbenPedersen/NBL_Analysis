import re
import pandas as pd
from typing import Dict
from src.advanced_stats import calculate_advanced_stats

def clean_up_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans up the statistics DataFrame."""
    if df.empty:
        return df
    df.columns = df.iloc[0].astype(str)
    df = df.drop(index=0).reset_index(drop=True)

    seen = {}
    new_columns = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
    df.columns = new_columns

    df["Name"] = df["Name"].astype(str)
    for col in df.columns:
        if col != "Name":
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def clean_up_available_players_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans up the available players DataFrame."""
    if df.empty:
        return df
    df.columns = ['Player', 'Overall Rating']
    for col in df.columns:
        if col != "Player":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def clean_up_draft_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans up the draft sheet DataFrame."""
    if df.empty:
        return df
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: "Round"})
    return df

def process_data(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Processes the raw data from Google Sheets into a single DataFrame
    for the Dash application.
    """
    # 1. Process available players from PG, SG, etc. sheets
    available_players_dfs = []
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    for pos in positions:
        sheet_name = f'draft_{pos}'
        if sheet_name in dataframes and not dataframes[sheet_name].empty:
            df = clean_up_available_players_df(dataframes[sheet_name])
            df['Position'] = pos
            available_players_dfs.append(df)
    
    available_players = pd.concat(available_players_dfs, ignore_index=True)
    available_players = available_players.rename(columns={'Player': 'Name'})

    # 2. Process drafted players from the "Draft Sheet"
    draft_sheet_raw = dataframes.get('draft_Draft Sheet')
    drafted_players_list = []
    if draft_sheet_raw is not None and not draft_sheet_raw.empty:
        draft_sheet = clean_up_draft_df(draft_sheet_raw)
        if 'Round' in draft_sheet.columns:
            melted_draft = draft_sheet.melt(
                id_vars="Round",
                var_name="Team",
                value_name="NameWithRating"
            )
            # Remove empty cells which get picked up as empty strings
            melted_draft = melted_draft[melted_draft['NameWithRating'].str.strip() != '']

            for _, row in melted_draft.iterrows():
                name_with_rating = row['NameWithRating']
                match = re.match(r'^(.*)\s\((\d+)\)$', name_with_rating)
                if match:
                    name = match.group(1).strip()
                    rating = int(match.group(2))
                    drafted_players_list.append({
                        'Name': name,
                        'Overall Rating': rating,
                        'Position': 'Unknown', # Position is not available for drafted players
                        'Draft Status': 'Drafted',
                        'Team': row['Team'],
                        'Round Picked': row['Round']
                    })
    
    drafted_players = pd.DataFrame(drafted_players_list)

    # 3. Combine available and drafted players
    available_players['Draft Status'] = 'Available'
    available_players['Team'] = 'None'
    available_players['Round Picked'] = 'Unpicked'

    all_players = pd.concat([available_players, drafted_players], ignore_index=True)

    # 4. Load stats and merge
    averages_df = clean_up_stats_df(dataframes.get('stats_Averages', pd.DataFrame()))
    
    merged_df = pd.merge(all_players, averages_df, on='Name', how='left')

    # 5. Calculate advanced stats
    merged_df = calculate_advanced_stats(merged_df)

    return merged_df 