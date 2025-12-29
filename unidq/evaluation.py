"""
Evaluation metrics for UNIDQ tasks
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, mean_squared_error, mean_absolute_error
)
from typing import Dict, List, Tuple, Optional


def evaluate_classification(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_name: str = "classification"
) -> Dict[str, float]:
    """
    Evaluate classification metrics.
    
    Args:
        predictions: Model predictions [batch_size, num_classes] or [batch_size]
        labels: Ground truth labels [batch_size]
        task_name: Name of the task
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if predictions.dim() > 1:
        preds = torch.argmax(predictions, dim=-1).cpu().numpy()
        probs = torch.softmax(predictions, dim=-1).cpu().numpy()
    else:
        preds = predictions.cpu().numpy()
        probs = None
    
    labels = labels.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary' if len(np.unique(labels)) == 2 else 'weighted'
    )
    
    metrics = {
        f'{task_name}_accuracy': accuracy,
        f'{task_name}_precision': precision,
        f'{task_name}_recall': recall,
        f'{task_name}_f1': f1,
    }
    
    # Add ROC-AUC for binary classification
    if probs is not None and len(np.unique(labels)) == 2:
        try:
            auc = roc_auc_score(labels, probs[:, 1])
            metrics[f'{task_name}_auc'] = auc
        except:
            pass
    
    return metrics


def evaluate_regression(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_name: str = "regression"
) -> Dict[str, float]:
    """
    Evaluate regression metrics.
    
    Args:
        predictions: Model predictions [batch_size] or [batch_size, 1]
        labels: Ground truth labels [batch_size]
        task_name: Name of the task
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy and flatten
    preds = predictions.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mse)
    
    metrics = {
        f'{task_name}_mse': mse,
        f'{task_name}_mae': mae,
        f'{task_name}_rmse': rmse,
    }
    
    return metrics


def evaluate_error_detection(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate error detection task.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    return evaluate_classification(predictions, labels, "error_detection")


def evaluate_imputation(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    is_categorical: bool = True
) -> Dict[str, float]:
    """
    Evaluate imputation task.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        is_categorical: Whether imputation is categorical or numerical
        
    Returns:
        Dictionary of metrics
    """
    if is_categorical:
        return evaluate_classification(predictions, labels, "imputation")
    else:
        return evaluate_regression(predictions, labels, "imputation")


def evaluate_duplicate_detection(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate duplicate detection task.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    return evaluate_classification(predictions, labels, "duplicate_detection")


def evaluate_outlier_detection(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate outlier detection task.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    return evaluate_classification(predictions, labels, "outlier_detection")


def evaluate_schema_matching(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate schema matching task.
    
    Args:
        predictions: Model predictions (similarity scores)
        labels: Ground truth labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert similarity scores to binary predictions
    binary_preds = (predictions > threshold).long()
    
    return evaluate_classification(binary_preds, labels, "schema_matching")


def evaluate_all_tasks(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    task_configs: Optional[Dict[str, dict]] = None
) -> Dict[str, float]:
    """
    Evaluate model on all tasks.
    
    Args:
        model: UNIDQ model
        dataloader: Data loader
        device: Device to run evaluation on
        task_configs: Task-specific configurations
        
    Returns:
        Dictionary of all metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Default task configurations
    if task_configs is None:
        task_configs = {
            'error_detection': {'type': 'classification'},
            'imputation': {'type': 'classification'},
            'duplicate_detection': {'type': 'classification'},
            'outlier_detection': {'type': 'classification'},
            'schema_matching': {'type': 'schema_matching'},
        }
    
    # Collect predictions and labels
    task_predictions = {task: [] for task in task_configs.keys()}
    task_labels = {task: [] for task in task_configs.keys()}
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            # Collect outputs for each task
            for task_name in task_configs.keys():
                if task_name in outputs:
                    label_key = f'{task_name}_labels'
                    if label_key in batch:
                        task_predictions[task_name].append(outputs[task_name])
                        task_labels[task_name].append(batch[label_key])
    
    # Concatenate predictions and labels
    for task_name in task_configs.keys():
        if task_predictions[task_name]:
            task_predictions[task_name] = torch.cat(task_predictions[task_name], dim=0)
            task_labels[task_name] = torch.cat(task_labels[task_name], dim=0)
    
    # Calculate metrics for each task
    all_metrics = {}
    
    for task_name, config in task_configs.items():
        if task_name not in task_predictions or len(task_predictions[task_name]) == 0:
            continue
        
        preds = task_predictions[task_name]
        labels = task_labels[task_name]
        
        # Aggregate sequence predictions if needed
        if preds.dim() == 3:  # [batch, seq_len, ...]
            preds = preds.mean(dim=1)
        if labels.dim() == 2:  # [batch, seq_len]
            labels = labels[:, 0]  # Take first token label as representative
        
        # Evaluate based on task type
        if config['type'] == 'classification':
            metrics = evaluate_classification(preds, labels, task_name)
        elif config['type'] == 'regression':
            metrics = evaluate_regression(preds, labels, task_name)
        elif config['type'] == 'schema_matching':
            metrics = evaluate_schema_matching(preds, labels)
        else:
            continue
        
        all_metrics.update(metrics)
    
    return all_metrics
