"""
Training utilities for UNIDQ
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm
import os


class UNIDQTrainer:
    """
    Trainer class for UNIDQ multi-task learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        task_weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: UNIDQ model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            optimizer: Optimizer (if None, uses Adam)
            task_weights: Weights for each task loss
            device: Device to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        
        # Task weights (equal by default)
        self.task_weights = task_weights or {
            'error_detection': 1.0,
            'imputation': 1.0,
            'schema_matching': 1.0,
            'duplicate_detection': 1.0,
            'outlier_detection': 1.0,
        }
        
        # Loss functions for different tasks
        self.loss_functions = {
            'error_detection': nn.CrossEntropyLoss(),
            'imputation': nn.CrossEntropyLoss(),
            'schema_matching': nn.MSELoss(),
            'duplicate_detection': nn.CrossEntropyLoss(),
            'outlier_detection': nn.CrossEntropyLoss(),
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses per task
        """
        self.model.train()
        epoch_losses = {task: 0.0 for task in self.task_weights.keys()}
        epoch_losses['total'] = 0.0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = self.model(input_ids, attention_mask)
            
            # Compute losses
            total_loss = 0.0
            batch_losses = {}
            
            for task_name, task_output in outputs.items():
                label_key = f'{task_name}_labels'
                
                if label_key in batch:
                    labels = batch[label_key]
                    loss_fn = self.loss_functions.get(task_name, nn.MSELoss())
                    
                    # Reshape for classification tasks
                    if task_name in ['error_detection', 'duplicate_detection', 'outlier_detection']:
                        # task_output: [batch, seq_len, num_classes]
                        # labels: [batch, seq_len] or [batch]
                        if labels.dim() == 1:
                            # Aggregate sequence predictions (use mean pooling)
                            task_output = task_output.mean(dim=1)
                        else:
                            # Flatten for sequence labeling
                            task_output = task_output.view(-1, task_output.size(-1))
                            labels = labels.view(-1)
                    
                    loss = loss_fn(task_output, labels)
                    weighted_loss = self.task_weights[task_name] * loss
                    total_loss += weighted_loss
                    
                    batch_losses[task_name] = loss.item()
                    epoch_losses[task_name] += loss.item()
            
            epoch_losses['total'] += total_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix(batch_losses)
        
        # Average losses
        num_batches = len(self.train_dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dictionary of average losses per task
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        eval_losses = {task: 0.0 for task in self.task_weights.keys()}
        eval_losses['total'] = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                outputs = self.model(input_ids, attention_mask)
                
                # Compute losses
                total_loss = 0.0
                
                for task_name, task_output in outputs.items():
                    label_key = f'{task_name}_labels'
                    
                    if label_key in batch:
                        labels = batch[label_key]
                        loss_fn = self.loss_functions.get(task_name, nn.MSELoss())
                        
                        # Reshape for classification tasks
                        if task_name in ['error_detection', 'duplicate_detection', 'outlier_detection']:
                            if labels.dim() == 1:
                                task_output = task_output.mean(dim=1)
                            else:
                                task_output = task_output.view(-1, task_output.size(-1))
                                labels = labels.view(-1)
                        
                        loss = loss_fn(task_output, labels)
                        weighted_loss = self.task_weights[task_name] * loss
                        total_loss += weighted_loss
                        
                        eval_losses[task_name] += loss.item()
                
                eval_losses['total'] += total_loss.item()
        
        # Average losses
        num_batches = len(self.val_dataloader)
        for key in eval_losses:
            eval_losses[key] /= num_batches
        
        return eval_losses
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_best: Whether to save the best model
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch()
            history['train_loss'].append(train_losses['total'])
            
            print(f"Training Loss: {train_losses['total']:.4f}")
            for task, loss in train_losses.items():
                if task != 'total':
                    print(f"  {task}: {loss:.4f}")
            
            # Evaluate
            if self.val_dataloader is not None:
                val_losses = self.evaluate()
                history['val_loss'].append(val_losses['total'])
                
                print(f"Validation Loss: {val_losses['total']:.4f}")
                for task, loss in val_losses.items():
                    if task != 'total':
                        print(f"  {task}: {loss:.4f}")
                
                # Save best model
                if save_best and val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        self.model.save_pretrained(os.path.join(save_dir, 'best_model'))
                        print(f"Saved best model to {save_dir}/best_model")
        
        return history
