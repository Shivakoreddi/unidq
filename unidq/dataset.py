"""
UNIDQ Dataset
=============

Dataset class for multi-task data quality training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Union


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task data quality training.
    
    Handles data for all 6 tasks:
    1. Error Detection
    2. Data Repair
    3. Missing Value Imputation
    4. Label Noise Detection
    5. Label Classification
    6. Data Valuation
    
    Parameters
    ----------
    dirty_features : np.ndarray
        Dirty/corrupted features, shape (n_samples, n_features)
    clean_features : np.ndarray, optional
        Clean/ground-truth features, shape (n_samples, n_features)
        If None, uses dirty_features
    error_mask : np.ndarray, optional
        Binary mask indicating errors, shape (n_samples, n_features)
        1 = error, 0 = clean
    missing_mask : np.ndarray, optional
        Binary mask indicating missing values, shape (n_samples, n_features)
    labels : np.ndarray, optional
        Observed (potentially noisy) labels, shape (n_samples,)
    clean_labels : np.ndarray, optional
        Ground-truth labels, shape (n_samples,)
        If None, uses labels
    noise_mask : np.ndarray, optional
        Binary mask indicating noisy labels, shape (n_samples,)
        1 = noisy, 0 = clean
        
    Example
    -------
    >>> dataset = MultiTaskDataset(
    ...     dirty_features=X_dirty,
    ...     clean_features=X_clean,
    ...     error_mask=errors,
    ...     labels=y_noisy,
    ...     clean_labels=y_clean
    ... )
    >>> batch = dataset[0]
    >>> print(batch.keys())
    """
    
    def __init__(
        self,
        dirty_features: np.ndarray,
        clean_features: Optional[np.ndarray] = None,
        error_mask: Optional[np.ndarray] = None,
        missing_mask: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        clean_labels: Optional[np.ndarray] = None,
        noise_mask: Optional[np.ndarray] = None,
        compute_z_scores: bool = True,
    ):
        # Convert to float32
        self.dirty = dirty_features.astype(np.float32)
        self.n_samples, self.n_features = self.dirty.shape
        
        # Clean features (default to dirty if not provided)
        if clean_features is not None:
            self.clean = clean_features.astype(np.float32)
        else:
            self.clean = self.dirty.copy()
        
        # Error mask (default to zeros)
        if error_mask is not None:
            self.error_mask = error_mask.astype(np.float32)
        else:
            self.error_mask = np.zeros((self.n_samples, self.n_features), dtype=np.float32)
        
        # Missing mask (default to zeros)
        if missing_mask is not None:
            self.missing_mask = missing_mask.astype(np.float32)
        else:
            self.missing_mask = np.zeros((self.n_samples, self.n_features), dtype=np.float32)
        
        # Labels (default to zeros)
        if labels is not None:
            self.labels = labels.astype(np.int64)
        else:
            self.labels = np.zeros(self.n_samples, dtype=np.int64)
        
        # Clean labels (default to observed labels)
        if clean_labels is not None:
            self.clean_labels = clean_labels.astype(np.int64)
        else:
            self.clean_labels = self.labels.copy()
        
        # Noise mask (default to zeros)
        if noise_mask is not None:
            self.noise_mask = noise_mask.astype(np.float32)
        else:
            self.noise_mask = np.zeros(self.n_samples, dtype=np.float32)
        
        # Compute z-scores for anomaly detection
        if compute_z_scores:
            self._compute_z_scores()
        else:
            self.z_scores = np.zeros_like(self.dirty)
        
        # Handle NaN values
        self._handle_nan()
        
        # Compute quality scores for data valuation
        self._compute_quality_scores()
    
    def _compute_z_scores(self):
        """Compute z-scores for each feature."""
        mean = np.nanmean(self.dirty, axis=0)
        std = np.nanstd(self.dirty, axis=0) + 1e-8
        self.z_scores = np.abs((self.dirty - mean) / std)
        self.z_scores = np.nan_to_num(self.z_scores, nan=0.0)
    
    def _handle_nan(self):
        """Replace NaN values with zeros."""
        self.dirty = np.nan_to_num(self.dirty, nan=0.0)
        self.clean = np.nan_to_num(self.clean, nan=0.0)
        self.z_scores = np.nan_to_num(self.z_scores, nan=0.0)
    
    def _compute_quality_scores(self):
        """Compute data quality scores for each sample."""
        # Error rate per sample
        error_rate = self.error_mask.mean(axis=1)
        
        # Missing rate per sample
        missing_rate = self.missing_mask.mean(axis=1)
        
        # Quality score: higher is better
        self.quality_scores = (1.0 - 0.5 * error_rate - 0.5 * missing_rate).astype(np.float32)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - dirty_features: Input features
            - z_scores: Z-scores for anomaly detection
            - error_mask: Binary error mask
            - clean_features: Ground-truth features
            - repair_mask: Mask for repair task (same as error_mask)
            - missing_mask: Binary missing value mask
            - impute_targets: Ground-truth values for imputation
            - dirty_label: Observed (potentially noisy) label
            - clean_label: Ground-truth label
            - noise_label: Binary indicating if label is noisy
            - quality_score: Data quality score
        """
        return {
            'dirty_features': torch.tensor(self.dirty[idx], dtype=torch.float32),
            'z_scores': torch.tensor(self.z_scores[idx], dtype=torch.float32),
            'error_mask': torch.tensor(self.error_mask[idx], dtype=torch.long),
            'clean_features': torch.tensor(self.clean[idx], dtype=torch.float32),
            'repair_mask': torch.tensor(self.error_mask[idx], dtype=torch.float32),
            'missing_mask': torch.tensor(self.missing_mask[idx], dtype=torch.float32),
            'impute_targets': torch.tensor(self.clean[idx], dtype=torch.float32),
            'dirty_label': torch.tensor(self.labels[idx], dtype=torch.long),
            'clean_label': torch.tensor(self.clean_labels[idx], dtype=torch.long),
            'noise_label': torch.tensor(int(self.noise_mask[idx]), dtype=torch.long),
            'quality_score': torch.tensor(self.quality_scores[idx], dtype=torch.float32),
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get dataset statistics.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - n_samples: Number of samples
            - n_features: Number of features
            - error_rate: Overall error rate
            - missing_rate: Overall missing rate
            - noise_rate: Label noise rate
        """
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'error_rate': float(self.error_mask.mean()),
            'missing_rate': float(self.missing_mask.mean()),
            'noise_rate': float(self.noise_mask.mean()),
        }
    
    @classmethod
    def from_dataframe(
        cls,
        dirty_df,
        clean_df=None,
        label_column: Optional[str] = None,
        **kwargs
    ) -> 'MultiTaskDataset':
        """
        Create dataset from pandas DataFrames.
        
        Parameters
        ----------
        dirty_df : pd.DataFrame
            Dirty/corrupted data
        clean_df : pd.DataFrame, optional
            Clean/ground-truth data
        label_column : str, optional
            Column name for labels
        **kwargs
            Additional arguments passed to __init__
            
        Returns
        -------
        MultiTaskDataset
        """
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        
        # Extract labels if specified
        labels = None
        clean_labels = None
        
        if label_column is not None:
            if label_column in dirty_df.columns:
                le = LabelEncoder()
                labels = le.fit_transform(dirty_df[label_column])
                dirty_df = dirty_df.drop(columns=[label_column])
                
                if clean_df is not None and label_column in clean_df.columns:
                    clean_labels = le.transform(clean_df[label_column])
                    clean_df = clean_df.drop(columns=[label_column])
        
        # Encode features
        n_samples = len(dirty_df)
        n_features = len(dirty_df.columns)
        
        dirty_encoded = np.zeros((n_samples, n_features), dtype=np.float32)
        clean_encoded = np.zeros((n_samples, n_features), dtype=np.float32) if clean_df is not None else None
        error_mask = np.zeros((n_samples, n_features), dtype=np.float32) if clean_df is not None else None
        
        for j, col in enumerate(dirty_df.columns):
            # Try numeric encoding first
            dirty_numeric = pd.to_numeric(dirty_df[col], errors='coerce')
            
            if dirty_numeric.notna().mean() > 0.5:
                # Numeric column
                mean = dirty_numeric.mean()
                std = dirty_numeric.std() + 1e-8
                dirty_encoded[:, j] = ((dirty_numeric.fillna(mean) - mean) / std).values
                
                if clean_df is not None:
                    clean_numeric = pd.to_numeric(clean_df[col], errors='coerce')
                    clean_encoded[:, j] = ((clean_numeric.fillna(mean) - mean) / std).values
            else:
                # Categorical column
                le = LabelEncoder()
                all_vals = dirty_df[col].fillna('__NA__').astype(str)
                if clean_df is not None:
                    all_vals = pd.concat([all_vals, clean_df[col].fillna('__NA__').astype(str)])
                le.fit(all_vals)
                
                dirty_encoded[:, j] = le.transform(dirty_df[col].fillna('__NA__').astype(str))
                max_val = max(dirty_encoded[:, j].max(), 1)
                dirty_encoded[:, j] /= max_val
                
                if clean_df is not None:
                    clean_encoded[:, j] = le.transform(clean_df[col].fillna('__NA__').astype(str)) / max_val
            
            # Compute error mask
            if clean_df is not None:
                d = dirty_df[col].fillna('').astype(str).str.strip().str.lower()
                c = clean_df[col].fillna('').astype(str).str.strip().str.lower()
                error_mask[:, j] = (d != c).values.astype(np.float32)
        
        return cls(
            dirty_features=dirty_encoded,
            clean_features=clean_encoded,
            error_mask=error_mask,
            labels=labels,
            clean_labels=clean_labels,
            **kwargs
        )
