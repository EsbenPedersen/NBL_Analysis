import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Optional, Dict


def generate_heatmap_style(df: DataFrame, column: str, n_bins: int = 7, colors: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """
    Generates conditional styling rules for a Dash DataTable column to create a heatmap effect.
    """
    if colors is None:
        colors = ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']

    styles: List[Dict[str, object]] = []
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return styles

    col_data = df[column].dropna()
    if col_data.empty:
        return styles

    min_val = col_data.min()
    max_val = col_data.max()

    if min_val == max_val:
        styles.append({
            'if': {'column_id': column},
            'backgroundColor': colors[len(colors) // 2],
            'color': 'white'
        })
        return styles

    bounds = np.linspace(min_val, max_val, n_bins + 1)

    for i in range(n_bins):
        min_bound = bounds[i]
        max_bound = bounds[i + 1]

        if i == n_bins - 1:
            filter_q = f'{{{column}}} >= {min_bound} && {{{column}}} <= {max_bound}'
        else:
            filter_q = f'{{{column}}} >= {min_bound} && {{{column}}} < {max_bound}'

        bg_color = colors[i]
        text_color = 'white' if i >= n_bins / 2 else 'black'

        styles.append({
            'if': {
                'filter_query': filter_q,
                'column_id': column
            },
            'backgroundColor': bg_color,
            'color': text_color
        })
    return styles


def create_slider_marks(min_val: float, max_val: float, num_marks: int = 10) -> Dict[int, str]:
    """Create evenly spaced, rounded marks for a slider."""
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return {0: '0'}
    step_values = np.linspace(min_val, max_val, num_marks)
    return {int(value): '{:.2f}'.format(value) for value in step_values}


def preprocess_size_data(size_data: pd.Series) -> pd.Series:
    """
    Normalize a numeric Series to a 10-50 range for Plotly marker sizes, handling NaNs gracefully.
    """
    if size_data.empty or size_data.isnull().all():
        return pd.Series(10, index=size_data.index)

    size_data = size_data.fillna(size_data.mean())
    if size_data.isnull().all():
        return pd.Series(10, index=size_data.index)

    min_val = size_data.min()
    if min_val < 0:
        size_data = size_data + abs(min_val)

    min_val = size_data.min()
    max_val = size_data.max()

    if pd.isna(max_val) or pd.isna(min_val) or max_val == min_val:
        return pd.Series(10, index=size_data.index)

    size_data = 10 + ((size_data - min_val) / (max_val - min_val)) * 40
    return size_data.fillna(10)


