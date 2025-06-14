import re
import pandas as pd
from typing import Dict
from src.advanced_stats import calculate_advanced_stats
from fuzzywuzzy import process as fuzzy_process, fuzz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_best_match(target: str, candidates: list[str], score_cutoff: int = 80) -> tuple[str | None, int]:
    """
    Gets the best fuzzy match for a target string from a list of candidates.
    Returns a tuple of (best_match, score).
    """
    from thefuzz import process
    
    # First try exact match after standardization
    std_target = _standardize_name(target)
    for candidate in candidates:
        if _standardize_name(candidate) == std_target:
            return candidate, 100
            
    # Then try partial matches
    for candidate in candidates:
        std_candidate = _standardize_name(candidate)
        # Check if target is a subset of candidate or vice versa
        if std_target in std_candidate or std_candidate in std_target:
            return candidate, 95
            
    # Finally try fuzzy matching
    result = process.extractOne(target, candidates, score_cutoff=score_cutoff)
    if result:
        return result[0], result[1]
    return None, 0

def clean_up_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up the statistics DataFrame by standardizing column names,
    parsing percentage values, and coercing data types.
    """
    if df.empty:
        return df

    df.columns = df.iloc[0].astype(str)
    df = df.drop(index=0).reset_index(drop=True)

    # 1. Basic cleanup and standardization of column names
    df.columns = [col.strip() for col in df.columns]
    column_mapping = {
        'Assist': 'Assists',
        'Steal': 'Steals',
        'Field Goal Per.': 'FG%',
        'Three Point Per.': '3pt%',
        'Free Throw Per.': 'FT%',
        'Effective FG': 'EFG%',
        'True Shooting': 'TS%',
        'A/TO Ratio': 'A/TO',
        'AST/TO': 'A/TO',
        'Player Efficency': 'PER',
        'Value Over Replacement Player': 'VORP',
        'FG Attempt': 'FG Attempted',
        '3FG Made': '3pt Made',
        '3FG Attempt': '3pt Attempted',
        'FT Attempt': 'FT Attempted',
    }
    df.rename(columns=column_mapping, inplace=True)

    # 2. Remove empty and duplicate columns
    if '' in df.columns:
        df = df.drop(columns=[''])
    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Handle percentage columns
    percentage_cols = ['FG%', '3pt%', 'FT%', 'EFG%', 'TS%']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0

    # 4. Coerce all other columns to their proper types
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str)

    for col in df.columns:
        if col != "Name" and col not in percentage_cols:
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

def _standardize_name(name: str) -> str:
    """
    Standardizes a name by sorting its parts alphabetically.
    This helps match names regardless of order (e.g., "Wei Ren" matches "Ren Wei").
    """
    if not name:
        return ""
    # Split on spaces and remove any empty parts
    parts = [p.strip().lower() for p in name.split() if p.strip()]
    # Sort parts alphabetically
    parts.sort()
    return ''.join(parts)

def process_data(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Processes the raw data from Google Sheets into a single DataFrame
    for the Dash application.
    """
    # 1. Load master player list from snapshot sheets
    snapshot_players_dfs = []
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    for pos in positions:
        sheet_name = f'snapshot_{pos}'
        if sheet_name in dataframes and not dataframes[sheet_name].empty:
            df = clean_up_available_players_df(dataframes[sheet_name].copy())
            df['Position'] = pos
            snapshot_players_dfs.append(df)

    if not snapshot_players_dfs:
        logging.warning("Snapshot sheets not found or empty. Falling back to legacy processing.")
        return _process_data_legacy(dataframes)

    master_player_list = pd.concat(snapshot_players_dfs, ignore_index=True)
    master_player_list = master_player_list.rename(columns={'Player': 'Name'})
    master_player_list['name_key'] = master_player_list['Name'].apply(_standardize_name)
    canonical_keys = list(master_player_list['name_key'].unique())

    # 2. Process Draft Sheet to get drafted players info
    draft_sheet_raw = dataframes.get('draft_Draft Sheet')
    drafted_players_list = []
    if draft_sheet_raw is not None and not draft_sheet_raw.empty:
        draft_sheet = clean_up_draft_df(draft_sheet_raw.copy())
        all_teams = [col for col in draft_sheet.columns if col != 'Round']
        
        # Create a mapping from GM first names to full team headers
        gm_map = {}
        for team_header in all_teams:
            gm_name = team_header.split('(')[0].strip()
            if gm_name:
                # Use full GM name for standardization
                std_gm_name = _standardize_name(gm_name)
                if std_gm_name not in gm_map:
                    gm_map[std_gm_name] = team_header

        melted_draft = draft_sheet.melt(id_vars="Round", var_name="OriginalTeam", value_name="CellText")
        melted_draft = melted_draft[melted_draft['CellText'].str.strip() != '']

        for _, row in melted_draft.iterrows():
            cell_text, original_team = row['CellText'].strip(), row['OriginalTeam']
            final_team, player_string = original_team, cell_text

            trade_match = re.match(r'To\s+([^:]+):\s*(.*)', cell_text, re.IGNORECASE)
            if trade_match:
                target_gm_name, player_string = trade_match.group(1).strip(), trade_match.group(2).strip()
                std_target_gm = _standardize_name(target_gm_name)
                
                # Try to find an exact match first
                resolved_team = gm_map.get(std_target_gm)
                if not resolved_team:
                    # Then try fuzzy matching
                    best_gm_match, score = _get_best_match(std_target_gm, list(gm_map.keys()), score_cutoff=85)
                    if best_gm_match:
                        final_team = gm_map[best_gm_match]
                    else:
                        logging.warning(f"Could not resolve traded team for GM '{target_gm_name}'. Using original team '{original_team}'.")

            player_name = re.match(r'^(.*?)\s*(?:\(\d+\))?$', player_string).group(1).strip() if player_string else ''
            if not player_name: continue

            drafted_players_list.append({
                'name_key': _standardize_name(player_name),
                'Original Name': player_name, 'Team': final_team,
                'Round Picked': row['Round'], 'Draft Status': 'Drafted'
            })
    drafted_players_df = pd.DataFrame(drafted_players_list)

    # 3. Process Stats Sheet
    averages_df = clean_up_stats_df(dataframes.get('stats_Averages', pd.DataFrame()))
    if not averages_df.empty:
        averages_df['name_key'] = averages_df['Name'].apply(_standardize_name)

    # 4. Map keys from drafted and stats dataframes to the canonical master list keys
    if not drafted_players_df.empty:
        drafted_players_df['canonical_key'] = drafted_players_df['name_key'].apply(
            lambda x: _get_best_match(x, canonical_keys)[0]
        )
    if not averages_df.empty:
        averages_df['canonical_key'] = averages_df['name_key'].apply(
            lambda x: _get_best_match(x, canonical_keys)[0]
        )

    # 5. Merge dataframes
    merged_df = master_player_list
    if not drafted_players_df.empty:
        draft_info_to_merge = drafted_players_df[['canonical_key', 'Team', 'Round Picked', 'Draft Status']].dropna(subset=['canonical_key']).drop_duplicates(subset=['canonical_key'])
        merged_df = pd.merge(merged_df, draft_info_to_merge, left_on='name_key', right_on='canonical_key', how='left')
        merged_df.drop(columns=['canonical_key'], inplace=True)

    if not averages_df.empty:
        stats_to_merge = averages_df.drop(columns=['Name', 'name_key']).dropna(subset=['canonical_key']).drop_duplicates(subset=['canonical_key'])
        merged_df = pd.merge(merged_df, stats_to_merge, left_on='name_key', right_on='canonical_key', how='left')
        merged_df.drop(columns=['canonical_key'], inplace=True, errors='ignore')

    merged_df['Draft Status'] = merged_df['Draft Status'].fillna('Available')
    merged_df['Team'] = merged_df['Team'].fillna('None')
    merged_df['Round Picked'] = merged_df['Round Picked'].fillna('Unpicked')
    merged_df.drop(columns=['name_key'], inplace=True, errors='ignore')
    
    # 6. Calculate advanced stats
    return calculate_advanced_stats(merged_df)

def _process_data_legacy(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Original data processing logic for when snapshot is not available.
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