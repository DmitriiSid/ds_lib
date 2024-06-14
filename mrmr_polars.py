import pandas as pd 
import polars as pl
import numpy as np

from sklearn.feature_selection import f_regression

def _f_score(df: pl.DataFrame, target: str, features: list[str]) -> np.ndarray:
    """
    Compute the F-score for each feature in relation to the target variable.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    target (str): The name of the target column.
    features (list[str]): The list of feature columns.

    Returns:
    np.ndarray: The F-scores for each feature.
    """
    # Convert Polars DataFrame to Pandas DataFrame for f_regression
    df_pandas = df.to_pandas()
    
    # Separate the features and target
    X = df_pandas[features]
    y = df_pandas[target]
    
    # Compute the F-scores and p-values
    f_scores, _ = f_regression(X, y)
    
    return f_scores

def mrmr_polars(df: pl.DataFrame, target: str, k: int) -> list[str]:
    """
    Perform Minimum Redundancy Maximum Relevance (mRMR) feature selection.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    target (str): The name of the target column.
    k (int): The number of features to select.

    Returns:
    list[str]: The list of selected features.
    """
    features = df.columns
    features.remove(target)

    f_scores = _f_score(df, target, features)

    df_scaled = df.select(features).with_columns((pl.col(f) - pl.col(f).mean()) / pl.col(f).std() for f in features)

    cumulating_sum = np.zeros(len(features))
    top_idx = np.argmax(f_scores)
    selected_features = [features[top_idx]]
    
    for j in range(1, k):
        argmax = -1
        current_max = -1
        last_selected = selected_features[-1]
        
        for i, f in enumerate(features):
            cumulating_sum[i] += np.abs((df_scaled.get_column(last_selected) * df_scaled.get_column(f)).mean())
            denominator = cumulating_sum[i] / j
            new_score = f_scores[i] / denominator
            
            if new_score > current_max:
                current_max = new_score
                argmax = i
        
        selected_features.append(features[argmax])

    return selected_features