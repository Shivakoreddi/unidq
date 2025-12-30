"""
UNIDQ Trainer
=============

Training utilities for the UNIDQ model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Optional, Dict, Callable, List, Union
import numpy as np
from tqdm import tqdm

from .model import UNIDQ
from .evaluation import evaluate_all_tasks


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for UNIDQ.
    
    Combines losses from all 6 tasks with configurable weights.
    
    Parameters
    ----------
    task_weights : Dict[str, float], optional
        Weights for each task loss. Default weights are tuned for
        balanced multi-task learning.
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        # Default task weights (tuned for balanced learning)
        self.task_weights = task_weights or {
            'error': 1.0,      # Primary task
            'repair': 0.5,     # Secondary task
            'impute': 0.5,     # Secondary task
            'label_clf': 0.3,  # Auxiliary task
            'noise': 0.5,      # Important for noisy data
            'value': 0.3,      # Auxiliary task
        }
        
        # Class weights for imbalanced tasks
        self.error_weights = None
        self.noise_weights = None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple:
        """
        Compute multi-task loss.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs
        batch : Dict[str, torch.Tensor]
            Batch data
        device : torch.device
            Device for computation
            
        Returns
        -------
        tuple
            (total_loss, dict of individual losses)
        """
        losses = {}
        
        # Initialize class weights on correct device
        if self.error_weights is None or self.error_weights.device != device:
            self.error_weights = torch.tensor([1.0, 3.0], device=device)
            self.noise_weights = torch.tensor([1.0, 4.0], device=device)
        
        # =====================================================================
        # Task 1: Error Detection (Weighted Cross-Entropy)
        # =====================================================================
        error_logits = outputs['error_logits']
        error_labels = batch['error_mask'].to(device)
        losses['error'] = nn.functional.cross_entropy(
            error_logits.reshape(-1, 2),
            error_labels.reshape(-1).long(),
            weight=self.error_weights
        )
        
        # =====================================================================
        # Task 2: Data Repair (MSE on erroneous cells only)
        # =====================================================================
        repair_pred = outputs['repair_pred']
        repair_target = batch['clean_features'].to(device)
        repair_mask = batch['repair_mask'].to(device)
        
        if repair_mask.sum() > 0:
            losses['repair'] = nn.functional.mse_loss(
                repair_pred * repair_mask,
                repair_target * repair_mask
            )
        else:
            losses['repair'] = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # Task 3: Missing Value Imputation (MSE on missing cells only)
        # =====================================================================
        impute_pred = outputs['impute_pred']
        impute_target = batch['impute_targets'].to(device)
        impute_mask = batch['missing_mask'].to(device)
        
        if impute_mask.sum() > 0:
            losses['impute'] = nn.functional.mse_loss(
                impute_pred * impute_mask,
                impute_target * impute_mask
            )
        else:
            losses['impute'] = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # Task 4: Label Classification (Cross-Entropy)
        # =====================================================================
        losses['label_clf'] = nn.functional.cross_entropy(
            outputs['label_logits'],
            batch['clean_label'].to(device)
        )
        
        # =====================================================================
        # Task 5: Label Noise Detection (Weighted Cross-Entropy)
        # =====================================================================
        losses['noise'] = nn.functional.cross_entropy(
            outputs['noise_logits'],
            batch['noise_label'].to(device),
            weight=self.noise_weights
        )
        
        # =====================================================================
        # Task 6: Data Valuation (MSE)
        # =====================================================================
        losses['value'] = nn.functional.mse_loss(
            outputs['value_pred'],
            batch['quality_score'].to(device)
        )
        
        # Compute weighted total loss
        total_loss = sum(self.task_weights[k] * v for k, v in losses.items())
        
        return total_loss, losses


class UNIDQTrainer:
    """
    Trainer for UNIDQ model.
    
    Handles training loop, validation, early stopping, and checkpointing.
    
    Parameters
    ----------
    model : UNIDQ
        Model to train
    device : torch.device, optional
        Device for training. Auto-detected if None.
    task_weights : Dict[str, float], optional
        Weights for multi-task loss
        
    Example
    -------
    >>> model = UNIDQ(n_features=14)
    >>> trainer = UNIDQTrainer(model)
    >>> trainer.fit(train_dataset, val_dataset, epochs=50)
    """
    
    def __init__(
        self,
        model: UNIDQ,
        device: Optional[torch.device] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = MultiTaskLoss(task_weights)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        self.best_state = None
        self.best_score = -float('inf')
    
    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Train the model.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset, optional
            Validation dataset for early stopping
        epochs : int, default=50
            Number of training epochs
        batch_size : int, default=64
            Batch size
        learning_rate : float, default=5e-4
            Learning rate
        weight_decay : float, default=0.01
            Weight decay for AdamW
        patience : int, default=10
            Early stopping patience
        verbose : bool, default=True
            Print progress
        checkpoint_path : str, optional
            Path to save best model
            
        Returns
        -------
        Dict[str, List]
            Training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )
        
        # Training loop
        no_improvement = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics'].append(val_metrics)
                
                # Early stopping based on combined score
                score = val_metrics.get('error_f1', 0) + 0.5 * val_metrics.get('noise_f1', 0)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improvement = 0
                    
                    if checkpoint_path:
                        self.model.save(checkpoint_path)
                else:
                    no_improvement += 1
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Error F1: {val_metrics.get('error_f1', 0):.4f}, "
                          f"Noise F1: {val_metrics.get('noise_f1', 0):.4f}")
                
                # Early stopping
                if no_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
            
            self.scheduler.step()
        
        # Load best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return self.history
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            # Move data to device
            features = batch['dirty_features'].to(self.device)
            z_scores = batch['z_scores'].to(self.device)
            dirty_labels = batch['dirty_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(features, z_scores, dirty_labels)
            
            # Compute loss
            loss, _ = self.loss_fn(outputs, batch, self.device)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _validate(self, dataloader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['dirty_features'].to(self.device)
                z_scores = batch['z_scores'].to(self.device)
                dirty_labels = batch['dirty_label'].to(self.device)
                
                outputs = self.model(features, z_scores, dirty_labels)
                loss, _ = self.loss_fn(outputs, batch, self.device)
                
                total_loss += loss.item()
                n_batches += 1
        
        # Compute metrics
        metrics = evaluate_all_tasks(self.model, dataloader, self.device)
        
        return total_loss / max(n_batches, 1), metrics
    
    def evaluate(self, dataset: Dataset, batch_size: int = 64) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to evaluate
        batch_size : int, default=64
            Batch size
            
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return evaluate_all_tasks(self.model, dataloader, self.device)


def cross_validate(
    model_class,
    dataset: Dataset,
    n_features: int,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    random_seed: int = 42,
    verbose: bool = True,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation.
    
    Parameters
    ----------
    model_class : type
        Model class (e.g., UNIDQ)
    dataset : Dataset
        Full dataset
    n_features : int
        Number of features
    n_folds : int, default=5
        Number of folds
    epochs : int, default=50
        Training epochs per fold
    batch_size : int, default=64
        Batch size
    random_seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress
    **model_kwargs
        Additional arguments for model initialization
        
    Returns
    -------
    Dict[str, List[float]]
        Cross-validation results for each metric
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        if verbose:
            print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Create subsets
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())
        
        # Initialize model and trainer
        model = model_class(n_features=n_features, **model_kwargs)
        trainer = UNIDQTrainer(model)
        
        # Train
        trainer.fit(
            train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )
        
        # Evaluate
        results = trainer.evaluate(val_ds)
        fold_results.append(results)
        
        if verbose:
            print(f"  Error F1: {results['error_f1']:.4f}, "
                  f"Noise F1: {results['noise_f1']:.4f}, "
                  f"Repair R²: {results['repair_r2']:.4f}")
    
    # Aggregate results
    all_metrics = {}
    for metric in fold_results[0].keys():
        values = [r[metric] for r in fold_results]
        all_metrics[metric] = {
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    if verbose:
        print("\n" + "="*50)
        print("Cross-Validation Results (Mean ± Std)")
        print("="*50)
        for metric, stats in all_metrics.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return all_metrics
