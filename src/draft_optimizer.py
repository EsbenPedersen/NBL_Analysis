import pandas as pd

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
        players_df['Turnovers_inv'] = players_df['Turnovers'].max() - players_df['Turnovers']
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
    
    return players_df.sort_values(by='Value', ascending=False)

def get_draft_recommendations(available_players: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Generates an ideal team and a prioritized draft board from available players.
    """
    if available_players.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 1. Calculate value scores for all available players
    scored_players = calculate_z_scores(available_players)
    
    # 2. Identify the top 2 players for each position (Ideal Team)
    ideal_team_list = []
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    for pos in positions:
        top_players = scored_players[scored_players['Position'] == pos].head(2)
        ideal_team_list.append(top_players)
        
    ideal_team = pd.concat(ideal_team_list).reset_index(drop=True)
    if 'Value' in ideal_team.columns:
        ideal_team['Value'] = ideal_team['Value'].round(3)
    
    # 3. Calculate "Value Over Next Available" (VONA) to prioritize picks
    draft_board_list = []
    for pos in positions:
        pos_players = scored_players[scored_players['Position'] == pos].reset_index(drop=True)
        if len(pos_players) > 2:
            # VONA is the value gap between this player and the first player NOT in the ideal team
            vona_score = pos_players.loc[1, 'Value'] - pos_players.loc[2, 'Value']
            # We assign this VONA to the top 2 players of this position
            for i in range(min(2, len(pos_players))):
                player_data = pos_players.loc[i].to_dict()
                player_data['VONA'] = vona_score
                draft_board_list.append(player_data)
        elif len(pos_players) > 0:
             # If there are 2 or fewer players, their scarcity is high. VONA is their own value.
            for i in range(len(pos_players)):
                player_data = pos_players.loc[i].to_dict()
                player_data['VONA'] = player_data['Value'] # High scarcity
                draft_board_list.append(player_data)

    draft_board = pd.DataFrame(draft_board_list)
    if 'Value' in draft_board.columns:
        draft_board['Value'] = draft_board['Value'].round(3)
    if 'VONA' in draft_board.columns:
        draft_board['VONA'] = draft_board['VONA'].round(3)

    draft_board = draft_board.sort_values(by='VONA', ascending=False).reset_index(drop=True)
    
    return ideal_team, draft_board 