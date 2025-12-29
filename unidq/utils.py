"""
Utility functions for UNIDQ
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import random


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def prepare_data_quality_labels(
    df: pd.DataFrame,
    error_column: Optional[str] = None,
    missing_column: Optional[str] = None,
    duplicate_column: Optional[str] = None,
    outlier_column: Optional[str] = None,
) -> Dict[str, pd.Series]:
    """
    Prepare task labels from a dataframe.
    
    Args:
        df: Input dataframe
        error_column: Column name containing error labels
        missing_column: Column name for missing value imputation
        duplicate_column: Column name for duplicate detection
        outlier_column: Column name for outlier detection
        
    Returns:
        Dictionary of task labels
    """
    task_labels = {}
    
    if error_column and error_column in df.columns:
        task_labels['error_detection'] = df[error_column]
    
    if missing_column and missing_column in df.columns:
        task_labels['imputation'] = df[missing_column]
    
    if duplicate_column and duplicate_column in df.columns:
        task_labels['duplicate_detection'] = df[duplicate_column]
    
    if outlier_column and outlier_column in df.columns:
        task_labels['outlier_detection'] = df[outlier_column]
    
    return task_labels


def create_synthetic_errors(
    df: pd.DataFrame,
    error_rate: float = 0.1,
    error_types: List[str] = ['typo', 'swap', 'missing']
) -> pd.DataFrame:
    """
    Create synthetic errors in a dataframe for testing.
    
    Args:
        df: Input dataframe
        error_rate: Fraction of cells to corrupt
        error_types: Types of errors to introduce
        
    Returns:
        Dataframe with synthetic errors
    """
    df_corrupted = df.copy()
    
    num_cells = df.shape[0] * df.shape[1]
    num_errors = int(num_cells * error_rate)
    
    for _ in range(num_errors):
        row_idx = random.randint(0, df.shape[0] - 1)
        col_idx = random.randint(0, df.shape[1] - 1)
        col_name = df.columns[col_idx]
        
        error_type = random.choice(error_types)
        
        if error_type == 'missing':
            df_corrupted.iloc[row_idx, col_idx] = np.nan
        elif error_type == 'typo' and isinstance(df_corrupted.iloc[row_idx, col_idx], str):
            original = str(df_corrupted.iloc[row_idx, col_idx])
            if len(original) > 0:
                pos = random.randint(0, len(original) - 1)
                corrupted = original[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + original[pos+1:]
                df_corrupted.iloc[row_idx, col_idx] = corrupted
        elif error_type == 'swap' and df.shape[0] > 1:
            other_row = random.randint(0, df.shape[0] - 1)
            df_corrupted.iloc[row_idx, col_idx], df_corrupted.iloc[other_row, col_idx] = \
                df_corrupted.iloc[other_row, col_idx], df_corrupted.iloc[row_idx, col_idx]
    
    return df_corrupted


def compute_data_quality_score(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray]
) -> float:
    """
    Compute overall data quality score from predictions.
    
    Args:
        df: Input dataframe
        predictions: Dictionary of task predictions
        
    Returns:
        Quality score between 0 and 1
    """
    scores = []
    
    # Error detection score (inverse of error rate)
    if 'error_detection' in predictions:
        error_rate = predictions['error_detection'].mean()
        scores.append(1 - error_rate)
    
    # Completeness score
    if 'imputation' in predictions:
        missing_rate = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(1 - missing_rate)
    
    # Duplicate score
    if 'duplicate_detection' in predictions:
        duplicate_rate = predictions['duplicate_detection'].mean()
        scores.append(1 - duplicate_rate)
    
    # Outlier score
    if 'outlier_detection' in predictions:
        outlier_rate = predictions['outlier_detection'].mean()
        scores.append(1 - outlier_rate)
    
    # Average all scores
    if scores:
        return np.mean(scores)
    else:
        return 0.0


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary for pretty printing.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name}: {value:.{precision}f}")
    return "\n".join(lines)


def save_predictions(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    format: str = 'csv'
):
    """
    Save predictions to file.
    
    Args:
        predictions: Dictionary of predictions
        output_path: Path to save file
        format: Output format ('csv', 'json', 'parquet')
    """
    df = pd.DataFrame(predictions)
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
