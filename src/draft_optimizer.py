import pandas as pd
import numpy as np

def calculate_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a 'Value' score for each player based on Z-scores of key stats.
    """
    players_df = df.copy()
    
    # Define stat categories for Z-score calculation.
    # We select a mix of traditional and advanced stats.
    # Turnovers are a "negative" stat, so we'll handle that later.
    z_score_cats = [
        'Overall Rating', 'Points', 'Rebounds', 'Assists', 'Stocks', 'A/TO', 
        'TS%', 'PER', 'VORP', 'PPM'
    ]
    
    # Invert 'Turnovers' so that a higher score is better.
    if 'Turnovers' in players_df.columns:
        players_df['Turnovers_inv'] = -players_df['Turnovers']
        z_score_cats.append('Turnovers_inv')

    # Calculate Z-scores for each category
    for cat in z_score_cats:
        if cat in players_df.columns:
            mean = players_df[cat].mean()
            std = players_df[cat].std()
            if std > 0:
                players_df[f'{cat}_zscore'] = (players_df[cat] - mean) / std
            else:
                players_df[f'{cat}_zscore'] = 0
    
    # Calculate total 'Value' score
    z_score_cols = [f'{cat}_zscore' for cat in z_score_cats if f'{cat}_zscore' in players_df.columns]
    players_df['Value'] = players_df[z_score_cols].sum(axis=1)
    
    return players_df

def get_draft_recommendations(available_players: pd.DataFrame, league_size: int = 12, roster_spots: int = 2) -> (pd.DataFrame, pd.DataFrame):
    """
    Generates draft recommendations using a Value Over Replacement Player (VORP) model.
    
    Args:
        available_players (pd.DataFrame): DataFrame of players available to be drafted.
        league_size (int): Number of teams in the league.
        roster_spots (int): Number of players drafted per position for VORP calculation.

    Returns:
        A tuple of two DataFrames: (ideal_team, draft_board).
    """
    if available_players.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 1. Calculate a baseline 'Value' for all players
    scored_players = calculate_z_scores(available_players)
    
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    
    # 2. Determine Replacement Level for each position
    replacement_levels = {}
    for pos in positions:
        pos_players = scored_players[scored_players['Position'] == pos].sort_values('Value', ascending=False)
        replacement_idx = min(len(pos_players) - 1, league_size * roster_spots)
        
        if not pos_players.empty:
            replacement_levels[pos] = pos_players.iloc[replacement_idx]['Value']
        else:
            replacement_levels[pos] = 0 # Default to 0 if no players for a position

    # 3. Calculate VORP for each player
    scored_players['VORP'] = scored_players.apply(
        lambda row: row['Value'] - replacement_levels.get(row['Position'], 0),
        axis=1
    )

    # 4. Generate the Ideal Team (Top 2 players by VORP at each position)
    ideal_team_list = []
    for pos in positions:
        top_players = scored_players[scored_players['Position'] == pos].sort_values('VORP', ascending=False).head(2)
        ideal_team_list.append(top_players)
    ideal_team = pd.concat(ideal_team_list).reset_index(drop=True)

    # 5. Calculate Value Over Next Available (VONA)
    vona_list = []
    for index, player in scored_players.iterrows():
        pos = player['Position']
        # Find the next best player at the same position
        next_best_players = scored_players[
            (scored_players['Position'] == pos) & 
            (scored_players['Name'] != player['Name'])
        ].sort_values('VORP', ascending=False)
        
        if not next_best_players.empty:
            next_best_vorp = next_best_players.iloc[0]['VORP']
            vona = player['VORP'] - next_best_vorp
        else:
            vona = player['VORP'] # If they are the only one, their VONA is their VORP
            
        player_data = player.to_dict()
        player_data['VONA'] = vona
        vona_list.append(player_data)

    draft_board = pd.DataFrame(vona_list)

    # Clean up and sort final dataframes
    for df in [ideal_team, draft_board]:
        for col in ['Value', 'VORP', 'VONA']:
            if col in df.columns:
                df[col] = df[col].round(3)

    draft_board = draft_board.sort_values(by='VONA', ascending=False).reset_index(drop=True)
    
    return ideal_team, draft_board.head(10) 