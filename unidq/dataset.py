"""
Multi-Task Dataset Implementation
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task data quality learning.
    
    Handles data loading and preprocessing for multiple data quality tasks.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        task_labels: Dict[str, pd.Series],
        tokenizer: Optional[callable] = None,
        max_length: int = 512,
    ):
        """
        Initialize the multi-task dataset.
        
        Args:
            data: Input dataframe
            task_labels: Dictionary mapping task names to label series
            tokenizer: Tokenization function
            max_length: Maximum sequence length
        """
        self.data = data
        self.task_labels = task_labels
        self.tokenizer = tokenizer or self._default_tokenizer
        self.max_length = max_length
        
        # Validate data
        assert len(data) > 0, "Dataset cannot be empty"
        for task_name, labels in task_labels.items():
            assert len(labels) == len(data), f"Label length mismatch for task: {task_name}"
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing input_ids, attention_mask, and task labels
        """
        # Get row data
        row = self.data.iloc[idx]
        
        # Convert row to text representation
        text = self._row_to_text(row)
        
        # Tokenize
        tokens = self.tokenizer(text, max_length=self.max_length)
        
        # Prepare output
        output = {
            'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long),
        }
        
        # Add task labels
        for task_name, labels in self.task_labels.items():
            label_value = labels.iloc[idx]
            
            # Handle different label types
            if isinstance(label_value, (int, np.integer)):
                output[f'{task_name}_labels'] = torch.tensor(label_value, dtype=torch.long)
            elif isinstance(label_value, (float, np.floating)):
                output[f'{task_name}_labels'] = torch.tensor(label_value, dtype=torch.float)
            else:
                # Try to convert to numeric
                try:
                    output[f'{task_name}_labels'] = torch.tensor(float(label_value), dtype=torch.float)
                except (ValueError, TypeError):
                    output[f'{task_name}_labels'] = torch.tensor(0, dtype=torch.long)
        
        return output
    
    def _row_to_text(self, row: pd.Series) -> str:
        """
        Convert a dataframe row to text representation.
        
        Args:
            row: Pandas series representing a row
            
        Returns:
            Text representation of the row
        """
        # Simple approach: concatenate column names and values
        parts = []
        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}: {val}")
            else:
                parts.append(f"{col}: [MISSING]")
        
        return " | ".join(parts)
    
    def _default_tokenizer(self, text: str, max_length: int) -> Dict[str, List[int]]:
        """
        Default tokenizer (character-level).
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Simple character-level tokenization
        char_ids = [ord(c) % 256 for c in text[:max_length]]
        
        # Pad or truncate
        if len(char_ids) < max_length:
            attention_mask = [1] * len(char_ids) + [0] * (max_length - len(char_ids))
            char_ids = char_ids + [0] * (max_length - len(char_ids))
        else:
            attention_mask = [1] * max_length
            char_ids = char_ids[:max_length]
        
        return {
            'input_ids': char_ids,
            'attention_mask': attention_mask,
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        # Get all keys from first sample
        keys = batch[0].keys()
        
        # Stack tensors for each key
        collated = {}
        for key in keys:
            collated[key] = torch.stack([sample[key] for sample in batch])
        
        return collated
