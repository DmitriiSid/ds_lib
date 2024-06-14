import pandas as pd
import numpy as np
import polars as pl
import category_encoders as ce
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif, f_regression
from typing import List
import time
import mrmr  # Make sure to install this package if you haven't already

def generate_synthetic_data(n_samples=300000, n_features=400, n_informative=50, n_categorical=50, random_state=42):
    """
    Generate synthetic data with numerical and categorical features.

    Parameters:
    n_samples (int): Number of samples to generate.
    n_features (int): Total number of numerical features.
    n_informative (int): Number of informative numerical features.
    n_categorical (int): Number of categorical features to add.
    random_state (int): Random state for reproducibility.

    Returns:
    pd.DataFrame: DataFrame containing synthetic data with numerical and categorical features.
    """
    # Generate numerical features using make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=0, n_clusters_per_class=1, random_state=random_state)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_features)])
    df['target'] = y
    
    # Add categorical features
    for i in range(n_categorical):
        df[f'cat_{i}'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    
    return df

def _f_score(df: pl.DataFrame, target: str, features: list[str], task: str) -> np.ndarray:
    """
    Compute the F-score for each feature in relation to the target variable.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    target (str): The name of the target column.
    features (list[str]): The list of feature columns.
    task (str): The type of task ('regression' or 'classification').

    Returns:
    np.ndarray: The F-scores for each feature.
    """
    # Convert Polars DataFrame to Pandas DataFrame for f_regression
    df_pandas = df.to_pandas()
    
    # Separate the features and target
    X = df_pandas[features]
    y = df_pandas[target]
    
    # Compute the F-scores and p-values
    if task == 'regression':
        f_scores, _ = f_regression(X, y)
    elif task == 'classification':
        f_scores, _ = f_classif(X, y)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'.")
    
    return f_scores

def encode_df(X, y, cat_features, cat_encoding):
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X

def mrmr_polars(df: pl.DataFrame, target: str, k: int, task: str, cat_features: list[str] = None, cat_encoding: str = 'leave_one_out') -> list[str]:
    """
    Perform Minimum Redundancy Maximum Relevance (mRMR) feature selection.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    target (str): The name of the target column.
    k (int): The number of features to select.
    task (str): The type of task ('regression' or 'classification').
    cat_features (list[str]): List of categorical features.
    cat_encoding (str): The method for encoding categorical features.

    Returns:
    list[str]: The list of selected features.
    """
    features = df.columns
    features.remove(target)

    if cat_features:
        df = pl.from_pandas(encode_df(df.to_pandas(), df[target].to_pandas(), cat_features, cat_encoding))
        features = [f for f in df.columns if f != target]

    f_scores = _f_score(df, target, features, task)

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

def _mrmr_feature_selection(data: pd.DataFrame,
                            target_df: pd.DataFrame,
                            categorical_features: List[str] = None,
                            mrmr_cat_encoding: str = "target",
                            num_of_features: int = 200) -> List[str]:
    if categorical_features is None: categorical_features = []

    if num_of_features > 0:
        print("Using MRMR to feature selection")
        selected_features = mrmr.mrmr_classif(
            X=data,
            y=target_df["target"],
            K=num_of_features,
            cat_encoding=mrmr_cat_encoding,
            cat_features=categorical_features)
    else:
        selected_features = list(data.columns)

    return selected_features

# Generate synthetic data
df = generate_synthetic_data()
categorical_features = [col for col in df.columns if col.startswith('cat_')]

# Convert to polars DataFrame for custom MRMR function
df_polars = pl.from_pandas(df)

# Split data and target
data = df.drop(columns=['target'])
target_df = df[['target']]

# Test custom MRMR function
start_time = time.time()
selected_features_custom = mrmr_polars(df_polars, 'target', 10, 'classification', cat_features=categorical_features)
end_time = time.time()
print(f"Custom MRMR selected features: {selected_features_custom}")
print(f"Time taken by custom MRMR: {end_time - start_time} seconds")

# Test MRMR package function
start_time = time.time()
selected_features_mrmr = _mrmr_feature_selection(data, target_df, categorical_features, num_of_features=10)
end_time = time.time()
print(f"MRMR package selected features: {selected_features_mrmr}")
print(f"Time taken by MRMR package: {end_time - start_time} seconds")
