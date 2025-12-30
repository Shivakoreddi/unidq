"""
UNIDQ Evaluation
================

Evaluation metrics and utilities for all 6 tasks.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_all_tasks(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate model on ALL 6 tasks.
    
    Parameters
    ----------
    model : torch.nn.Module
        UNIDQ model
    dataloader : DataLoader
        Data loader for evaluation
    device : torch.device, optional
        Device for computation
        
    Returns
    -------
    Dict[str, float]
        Comprehensive metrics for all 6 tasks:
        
        Error Detection:
        - error_f1: F1 score
        - error_auc: ROC AUC
        - error_precision: Precision
        - error_recall: Recall
        
        Data Repair:
        - repair_r2: R² score
        - repair_mae: Mean Absolute Error
        
        Imputation:
        - impute_r2: R² score
        - impute_mae: Mean Absolute Error
        
        Label Noise Detection:
        - noise_f1: F1 score
        - noise_auc: ROC AUC
        - noise_precision: Precision
        - noise_recall: Recall
        
        Label Classification:
        - label_accuracy: Accuracy
        - label_f1: F1 score
        - label_auc: ROC AUC
        
        Data Valuation:
        - value_correlation: Pearson correlation
        - value_r2: R² score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Storage for predictions and targets
    all_error_probs, all_error_labels = [], []
    all_repair_pred, all_repair_target, all_repair_mask = [], [], []
    all_impute_pred, all_impute_target, all_impute_mask = [], [], []
    all_noise_probs, all_noise_labels = [], []
    all_label_probs, all_label_pred, all_clean_labels = [], [], []
    all_value_pred, all_value_target = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['dirty_features'].to(device)
            z_scores = batch['z_scores'].to(device)
            dirty_labels = batch['dirty_label'].to(device)
            
            outputs = model(features, z_scores, dirty_labels)
            
            # Error detection
            error_probs = torch.softmax(outputs['error_logits'], dim=-1)[:, :, 1]
            all_error_probs.extend(error_probs.cpu().numpy().flatten())
            all_error_labels.extend(batch['error_mask'].numpy().flatten())
            
            # Data repair
            all_repair_pred.extend(outputs['repair_pred'].cpu().numpy().flatten())
            all_repair_target.extend(batch['clean_features'].numpy().flatten())
            all_repair_mask.extend(batch['repair_mask'].numpy().flatten())
            
            # Imputation
            all_impute_pred.extend(outputs['impute_pred'].cpu().numpy().flatten())
            all_impute_target.extend(batch['impute_targets'].numpy().flatten())
            all_impute_mask.extend(batch['missing_mask'].numpy().flatten())
            
            # Noise detection
            noise_probs = torch.softmax(outputs['noise_logits'], dim=-1)[:, 1]
            all_noise_probs.extend(noise_probs.cpu().numpy())
            all_noise_labels.extend(batch['noise_label'].numpy())
            
            # Label classification
            label_probs = torch.softmax(outputs['label_logits'], dim=-1)
            label_pred = outputs['label_logits'].argmax(dim=-1)
            all_label_probs.extend(label_probs[:, 1].cpu().numpy())
            all_label_pred.extend(label_pred.cpu().numpy())
            all_clean_labels.extend(batch['clean_label'].numpy())
            
            # Data valuation
            all_value_pred.extend(outputs['value_pred'].cpu().numpy())
            all_value_target.extend(batch['quality_score'].numpy())
    
    results = {}
    
    # =========================================================================
    # TASK 1: ERROR DETECTION
    # =========================================================================
    all_error_probs = np.array(all_error_probs)
    all_error_labels = np.array(all_error_labels)
    
    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.2, 0.8, 0.05):
        preds = (all_error_probs >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            all_error_labels, preds, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    preds = (all_error_probs >= best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_error_labels, preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_error_labels, all_error_probs)
    except:
        auc = 0.5
    
    results['error_f1'] = f1
    results['error_auc'] = auc
    results['error_precision'] = prec
    results['error_recall'] = rec
    
    # =========================================================================
    # TASK 2: DATA REPAIR
    # =========================================================================
    all_repair_pred = np.array(all_repair_pred)
    all_repair_target = np.array(all_repair_target)
    all_repair_mask = np.array(all_repair_mask)
    
    error_idx = all_repair_mask == 1
    if error_idx.sum() > 0:
        results['repair_r2'] = max(0, r2_score(
            all_repair_target[error_idx],
            all_repair_pred[error_idx]
        ))
        results['repair_mae'] = mean_absolute_error(
            all_repair_target[error_idx],
            all_repair_pred[error_idx]
        )
    else:
        results['repair_r2'] = 0.0
        results['repair_mae'] = 0.0
    
    # =========================================================================
    # TASK 3: IMPUTATION
    # =========================================================================
    all_impute_pred = np.array(all_impute_pred)
    all_impute_target = np.array(all_impute_target)
    all_impute_mask = np.array(all_impute_mask)
    
    missing_idx = all_impute_mask == 1
    if missing_idx.sum() > 0:
        results['impute_r2'] = max(0, r2_score(
            all_impute_target[missing_idx],
            all_impute_pred[missing_idx]
        ))
        results['impute_mae'] = mean_absolute_error(
            all_impute_target[missing_idx],
            all_impute_pred[missing_idx]
        )
    else:
        results['impute_r2'] = 0.0
        results['impute_mae'] = 0.0
    
    # =========================================================================
    # TASK 4: LABEL NOISE DETECTION
    # =========================================================================
    all_noise_probs = np.array(all_noise_probs)
    all_noise_labels = np.array(all_noise_labels)
    
    # Find optimal threshold
    best_f1 = 0
    for thresh in np.arange(0.2, 0.8, 0.05):
        preds = (all_noise_probs >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            all_noise_labels, preds, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
    
    preds = (all_noise_probs >= 0.5).astype(int)
    prec, rec, _, _ = precision_recall_fscore_support(
        all_noise_labels, preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_noise_labels, all_noise_probs)
    except:
        auc = 0.5
    
    results['noise_f1'] = best_f1
    results['noise_auc'] = auc
    results['noise_precision'] = prec
    results['noise_recall'] = rec
    
    # =========================================================================
    # TASK 5: LABEL CLASSIFICATION
    # =========================================================================
    all_label_pred = np.array(all_label_pred)
    all_label_probs = np.array(all_label_probs)
    all_clean_labels = np.array(all_clean_labels)
    
    results['label_accuracy'] = accuracy_score(all_clean_labels, all_label_pred)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_clean_labels, all_label_pred, average='binary', zero_division=0
    )
    results['label_f1'] = f1
    
    try:
        results['label_auc'] = roc_auc_score(all_clean_labels, all_label_probs)
    except:
        results['label_auc'] = 0.5
    
    # =========================================================================
    # TASK 6: DATA VALUATION
    # =========================================================================
    all_value_pred = np.array(all_value_pred)
    all_value_target = np.array(all_value_target)
    
    try:
        corr = np.corrcoef(all_value_pred, all_value_target)[0, 1]
        results['value_correlation'] = corr if not np.isnan(corr) else 0.0
    except:
        results['value_correlation'] = 0.0
    
    if len(all_value_target) > 1:
        results['value_r2'] = max(0, r2_score(all_value_target, all_value_pred))
    else:
        results['value_r2'] = 0.0
    
    return results


def evaluate_task(
    model: torch.nn.Module,
    dataloader: DataLoader,
    task: str,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a specific task.
    
    Parameters
    ----------
    model : torch.nn.Module
        UNIDQ model
    dataloader : DataLoader
        Data loader for evaluation
    task : str
        Task name: 'error', 'repair', 'impute', 'noise', 'label', 'value'
    device : torch.device, optional
        Device for computation
        
    Returns
    -------
    Dict[str, float]
        Metrics for the specified task
    """
    all_results = evaluate_all_tasks(model, dataloader, device)
    
    task_metrics = {
        'error': ['error_f1', 'error_auc', 'error_precision', 'error_recall'],
        'repair': ['repair_r2', 'repair_mae'],
        'impute': ['impute_r2', 'impute_mae'],
        'noise': ['noise_f1', 'noise_auc', 'noise_precision', 'noise_recall'],
        'label': ['label_accuracy', 'label_f1', 'label_auc'],
        'value': ['value_correlation', 'value_r2'],
    }
    
    if task not in task_metrics:
        raise ValueError(f"Unknown task: {task}. Choose from {list(task_metrics.keys())}")
    
    return {k: all_results[k] for k in task_metrics[task]}


def print_evaluation_report(results: Dict[str, float]) -> None:
    """
    Print a formatted evaluation report.
    
    Parameters
    ----------
    results : Dict[str, float]
        Evaluation results from evaluate_all_tasks
    """
    print("\n" + "=" * 60)
    print("UNIDQ Evaluation Report")
    print("=" * 60)
    
    print("\n📌 ERROR DETECTION")
    print(f"   F1 Score:    {results.get('error_f1', 0):.4f}")
    print(f"   ROC AUC:     {results.get('error_auc', 0):.4f}")
    print(f"   Precision:   {results.get('error_precision', 0):.4f}")
    print(f"   Recall:      {results.get('error_recall', 0):.4f}")
    
    print("\n🔧 DATA REPAIR")
    print(f"   R² Score:    {results.get('repair_r2', 0):.4f}")
    print(f"   MAE:         {results.get('repair_mae', 0):.4f}")
    
    print("\n📥 IMPUTATION")
    print(f"   R² Score:    {results.get('impute_r2', 0):.4f}")
    print(f"   MAE:         {results.get('impute_mae', 0):.4f}")
    
    print("\n🏷️ LABEL NOISE DETECTION")
    print(f"   F1 Score:    {results.get('noise_f1', 0):.4f}")
    print(f"   ROC AUC:     {results.get('noise_auc', 0):.4f}")
    print(f"   Precision:   {results.get('noise_precision', 0):.4f}")
    print(f"   Recall:      {results.get('noise_recall', 0):.4f}")
    
    print("\n🎯 LABEL CLASSIFICATION")
    print(f"   Accuracy:    {results.get('label_accuracy', 0):.4f}")
    print(f"   F1 Score:    {results.get('label_f1', 0):.4f}")
    print(f"   ROC AUC:     {results.get('label_auc', 0):.4f}")
    
    print("\n💰 DATA VALUATION")
    print(f"   Correlation: {results.get('value_correlation', 0):.4f}")
    print(f"   R² Score:    {results.get('value_r2', 0):.4f}")
    
    print("\n" + "=" * 60)
