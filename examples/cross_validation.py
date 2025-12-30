#!/usr/bin/env python3
"""
UNIDQ Cross-Validation Example
==============================

5-Fold Cross-Validation for all 6 data quality tasks.
This example shows how to reproduce paper results.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer
from unidq.evaluation import evaluate_all_tasks, print_evaluation_report

# Configuration
N_FOLDS = 5
N_EPOCHS = 50
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


def load_raha_dataset(dataset_name: str, base_dir: str = './data') -> dict:
    """
    Load dataset in Raha format (dirty.csv + clean.csv).
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset (e.g., 'beers', 'flights', 'rayyan')
    base_dir : str
        Base directory containing datasets
        
    Returns
    -------
    dict
        Dictionary with encoded features, masks, and labels
    """
    path = os.path.join(base_dir, f'Raha/datasets/{dataset_name}')
    
    # Load CSV files
    dirty_df = pd.read_csv(f'{path}/dirty.csv', dtype=str, keep_default_na=False)
    clean_df = pd.read_csv(f'{path}/clean.csv', dtype=str, keep_default_na=False)
    
    # Align columns
    if list(dirty_df.columns) != list(clean_df.columns):
        if len(dirty_df.columns) == len(clean_df.columns):
            clean_df.columns = dirty_df.columns
    
    n_samples = min(len(dirty_df), len(clean_df))
    n_cols = len(dirty_df.columns)
    dirty_df = dirty_df.iloc[:n_samples]
    clean_df = clean_df.iloc[:n_samples]
    
    # Compute error mask (cell-level)
    error_mask = np.zeros((n_samples, n_cols), dtype=np.float32)
    for j, col in enumerate(dirty_df.columns):
        d = dirty_df[col].fillna('').astype(str).str.strip().str.lower()
        c = clean_df[col].fillna('').astype(str).str.strip().str.lower()
        error_mask[:, j] = (d != c).values.astype(np.float32)
    
    # Encode features
    dirty_encoded = np.zeros((n_samples, n_cols), dtype=np.float32)
    clean_encoded = np.zeros((n_samples, n_cols), dtype=np.float32)
    
    for j, col in enumerate(dirty_df.columns):
        dirty_numeric = pd.to_numeric(dirty_df[col], errors='coerce')
        clean_numeric = pd.to_numeric(clean_df[col], errors='coerce')
        
        if dirty_numeric.notna().mean() > 0.5:
            # Numeric column: standardize
            mean = dirty_numeric.mean()
            std = dirty_numeric.std() + 1e-8
            dirty_encoded[:, j] = ((dirty_numeric.fillna(mean) - mean) / std).values
            clean_encoded[:, j] = ((clean_numeric.fillna(mean) - mean) / std).values
        else:
            # Categorical column: label encode and normalize
            le = LabelEncoder()
            all_vals = pd.concat([
                dirty_df[col].fillna('__NA__').astype(str),
                clean_df[col].fillna('__NA__').astype(str)
            ])
            le.fit(all_vals)
            dirty_encoded[:, j] = le.transform(dirty_df[col].fillna('__NA__').astype(str))
            clean_encoded[:, j] = le.transform(clean_df[col].fillna('__NA__').astype(str))
            max_val = max(dirty_encoded[:, j].max(), 1)
            dirty_encoded[:, j] /= max_val
            clean_encoded[:, j] /= max_val
    
    # Create labels using PCA (unsupervised)
    np.random.seed(RANDOM_SEED)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(clean_encoded).flatten()
    clean_labels = (pc1 > np.median(pc1)).astype(np.int64)
    
    # Inject label noise (15%)
    noisy_labels = clean_labels.copy()
    noise_mask = np.zeros(n_samples, dtype=np.float32)
    n_noisy = int(n_samples * 0.15)
    noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
    noisy_labels[noisy_idx] = 1 - noisy_labels[noisy_idx]
    noise_mask[noisy_idx] = 1.0
    
    # Create missing mask (10% of clean cells)
    missing_mask = np.zeros((n_samples, n_cols), dtype=np.float32)
    n_missing = int(n_samples * n_cols * 0.1)
    available = np.where(error_mask == 0)
    if len(available[0]) > 0:
        n_missing = min(n_missing, len(available[0]))
        selected = np.random.choice(len(available[0]), n_missing, replace=False)
        for idx in selected:
            missing_mask[available[0][idx], available[1][idx]] = 1.0
    
    return {
        'dirty_encoded': dirty_encoded,
        'clean_encoded': clean_encoded,
        'error_mask': error_mask,
        'missing_mask': missing_mask,
        'noisy_labels': noisy_labels,
        'clean_labels': clean_labels,
        'noise_mask': noise_mask,
        'n_samples': n_samples,
        'n_features': n_cols,
    }


def run_cross_validation(dataset_name: str, data: dict, n_folds: int = 5, n_epochs: int = 50):
    """
    Run k-fold cross-validation for ALL 6 tasks.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset for display
    data : dict
        Dictionary from load_raha_dataset
    n_folds : int
        Number of folds
    n_epochs : int
        Training epochs per fold
        
    Returns
    -------
    tuple
        (summary_dict, list_of_fold_results)
    """
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION: {dataset_name} ({n_folds}-Fold)")
    print(f"{'='*70}")
    print(f"Samples: {data['n_samples']}, Features: {data['n_features']}")
    print(f"Error rate: {data['error_mask'].mean()*100:.1f}%")
    
    # Create full dataset
    dataset = MultiTaskDataset(
        dirty_features=data['dirty_encoded'],
        clean_features=data['clean_encoded'],
        error_mask=data['error_mask'],
        missing_mask=data['missing_mask'],
        labels=data['noisy_labels'],
        clean_labels=data['clean_labels'],
        noise_mask=data['noise_mask'],
    )
    
    # K-Fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Create subsets
        train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
        
        # Initialize model
        model = UNIDQ(
            n_features=data['n_features'],
            d_model=128,
            n_heads=4,
            n_layers=3,
        )
        
        # Train
        trainer = UNIDQTrainer(model, device=DEVICE)
        trainer.fit(
            train_ds, val_ds,
            epochs=n_epochs,
            batch_size=64,
            verbose=False,
        )
        
        # Evaluate
        results = trainer.evaluate(val_ds)
        fold_results.append(results)
        
        # Print fold results
        print(f"  Error F1:     {results['error_f1']:.4f}")
        print(f"  Noise F1:     {results['noise_f1']:.4f}")
        print(f"  Repair R²:    {results['repair_r2']:.4f}")
        print(f"  Impute R²:    {results['impute_r2']:.4f}")
        print(f"  Label Acc:    {results['label_accuracy']:.4f}")
        print(f"  Value Corr:   {results['value_correlation']:.4f}")
    
    # Aggregate results
    metrics = list(fold_results[0].keys())
    summary = {'dataset': dataset_name}
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name} (Mean ± Std)")
    print(f"{'='*70}")
    
    for metric in metrics:
        values = [r.get(metric, 0) for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        summary[f'{metric}_mean'] = mean
        summary[f'{metric}_std'] = std
        
        display_name = metric.replace('_', ' ').title()
        print(f"{display_name:25s}: {mean:.4f} ± {std:.4f}")
    
    return summary, fold_results


def run_all_datasets(datasets: list = None, base_dir: str = './data'):
    """
    Run cross-validation on multiple datasets.
    
    Parameters
    ----------
    datasets : list
        List of dataset names
    base_dir : str
        Base directory for data
        
    Returns
    -------
    tuple
        (list_of_summaries, dict_of_fold_results)
    """
    if datasets is None:
        datasets = ['beers', 'flights', 'rayyan']
    
    print("=" * 80)
    print("UNIDQ COMPREHENSIVE CROSS-VALIDATION - ALL 6 TASKS")
    print("=" * 80)
    
    all_summaries = []
    all_fold_results = {}
    
    for name in datasets:
        try:
            data = load_raha_dataset(name, base_dir)
            summary, fold_results = run_cross_validation(name, data, N_FOLDS, N_EPOCHS)
            all_summaries.append(summary)
            all_fold_results[name] = fold_results
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final summary table
    if all_summaries:
        print("\n" + "=" * 100)
        print("FINAL CROSS-VALIDATION RESULTS (ALL 6 TASKS)")
        print("=" * 100)
        
        # Table header
        header = f"{'Task':<25}"
        for s in all_summaries:
            header += f" | {s['dataset']:^20}"
        header += f" | {'Average':^20}"
        print(header)
        print("-" * len(header))
        
        # Metrics to display
        task_metrics = [
            ('Error Detection (F1)', 'error_f1'),
            ('Error Detection (AUC)', 'error_auc'),
            ('Data Repair (R²)', 'repair_r2'),
            ('Imputation (R²)', 'impute_r2'),
            ('Label Noise (F1)', 'noise_f1'),
            ('Label Noise (AUC)', 'noise_auc'),
            ('Label Clf (Acc)', 'label_accuracy'),
            ('Label Clf (F1)', 'label_f1'),
            ('Data Valuation (ρ)', 'value_correlation'),
        ]
        
        for task_name, metric in task_metrics:
            row = f"{task_name:<25}"
            values = []
            for s in all_summaries:
                mean = s.get(f'{metric}_mean', 0)
                std = s.get(f'{metric}_std', 0)
                row += f" | {mean:.3f}±{std:.3f}".center(20)
                values.append(mean)
            
            avg_mean = np.mean(values)
            row += f" | {avg_mean:.3f}".center(20)
            print(row)
        
        # Save results to CSV
        df = pd.DataFrame(all_summaries)
        df.to_csv('cross_validation_results.csv', index=False)
        print(f"\nResults saved to cross_validation_results.csv")
    
    return all_summaries, all_fold_results


if __name__ == "__main__":
    # Mount Google Drive if in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        BASE_DIR = '/content/drive/MyDrive/UNIDQ/data'
    except:
        BASE_DIR = './data'
    
    # Run on all datasets
    summaries, fold_results = run_all_datasets(
        datasets=['beers', 'flights', 'rayyan'],
        base_dir=BASE_DIR
    )
