import pandas as pd
import numpy as np

def get_col(df: pd.DataFrame, names: list) -> pd.Series:
    """Gets a column by trying multiple possible names."""
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(0, index=df.index)

def col_exists(df: pd.DataFrame, names: list) -> bool:
    """Checks if a column exists under any of the possible names."""
    return any(name in df.columns for name in names)

def calculate_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various advanced basketball statistics for a given DataFrame of player stats.
    The DataFrame should contain standardized column names.
    It will only calculate stats if they are not already present.
    """
    stats_df = df.copy()

    # Define required columns for calculations. We expect them to be standardized.
    required_cols = [
        'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Minutes',
        'FG Made', 'FG Attempted', '3pt Made', '3pt Attempted', 'FT Made', 'FT Attempted', 'Fouls'
    ]
    # Ensure required columns exist and are numeric, otherwise create them as 0
    for col in required_cols:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        else:
            stats_df[col] = 0

    # Replace inf/-inf with NaN, then fill NaN with 0 to prevent errors in calculations
    stats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stats_df.fillna(0, inplace=True)

    # Advanced Stat Calculations - only if they don't already exist
    if 'Stocks' not in stats_df.columns:
        stats_df['Stocks'] = stats_df['Steals'] + stats_df['Blocks']
    
    if 'A/TO' not in stats_df.columns:
        stats_df['A/TO'] = stats_df['Assists'] / stats_df['Turnovers'].replace(0, 1)

    if 'FG%' not in stats_df.columns:
        stats_df['FG%'] = stats_df['FG Made'] / stats_df['FG Attempted'].replace(0, 1)

    if '3pt%' not in stats_df.columns:
        stats_df['3pt%'] = stats_df['3pt Made'] / stats_df['3pt Attempted'].replace(0, 1)

    if 'FT%' not in stats_df.columns:
        stats_df['FT%'] = stats_df['FT Made'] / stats_df['FT Attempted'].replace(0, 1)

    if 'TS%' not in stats_df.columns:
        stats_df['TS%'] = stats_df['Points'] / (2 * (stats_df['FG Attempted'] + 0.44 * stats_df['FT Attempted'])).replace(0, 1)

    if 'EFG%' not in stats_df.columns:
        stats_df['EFG%'] = (stats_df['FG Made'] + 0.5 * stats_df['3pt Made']) / stats_df['FG Attempted'].replace(0, 1)
    
    if 'PER' not in stats_df.columns:
        fg_part = stats_df['FG Made'] * 85.91
        steals_part = stats_df['Steals'] * 53.897
        threes_part = stats_df['3pt Made'] * 51.757
        ft_part = stats_df['FT Made'] * 46.845
        blocks_part = stats_df['Blocks'] * 39.19
        # Note: OReb is not in the new data, so we can't calculate it here. Using Rebounds instead.
        oreb_part = stats_df['Rebounds'] * 0.5 * 39.19 # Assuming 50% of rebounds are offensive as a proxy
        assists_part = stats_df['Assists'] * 34.677
        dreb_part = stats_df['Rebounds'] * 0.5 * 14.707
        fouls_part = stats_df['Fouls'] * 17.174
        ft_miss_part = (stats_df['FT Attempted'] - stats_df['FT Made']) * 20.091
        fg_miss_part = (stats_df['FG Attempted'] - stats_df['FG Made']) * 39.19
        turnovers_part = stats_df['Turnovers'] * 53.897
        
        per_unadjusted = (fg_part + steals_part + threes_part + ft_part + blocks_part + 
                          oreb_part + assists_part + dreb_part - fouls_part - 
                          ft_miss_part - fg_miss_part - turnovers_part)
        
        stats_df['PER'] = per_unadjusted / stats_df['Minutes'].replace(0, 1)

    if 'VORP' not in stats_df.columns:
        stats_df['VORP'] = (stats_df['PER'] - 11.0) * (stats_df['Minutes'] / 48) # 11.0 is a common baseline for PER

    if 'PPM' not in stats_df.columns:
        stats_df['PPM'] = stats_df['Points'] / stats_df['Minutes'].replace(0, 1)

    # Clean up final dataframe again
    stats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stats_df.fillna(0, inplace=True)

    return stats_df 