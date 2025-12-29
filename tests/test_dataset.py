"""
Unit tests for MultiTaskDataset
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unidq.dataset import MultiTaskDataset


class TestMultiTaskDataset:
    """Test dataset class."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40],
            'city': ['NYC', 'LA', 'Chicago', 'Houston'],
            'salary': [50000, 60000, 70000, 80000],
        })
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample task labels."""
        return {
            'error_detection': pd.Series([0, 1, 0, 0]),
            'duplicate_detection': pd.Series([0, 0, 1, 0]),
            'outlier_detection': pd.Series([0, 0, 0, 1]),
        }
    
    def test_dataset_initialization(self, sample_dataframe, sample_labels):
        """Test dataset initialization."""
        dataset = MultiTaskDataset(
            data=sample_dataframe,
            task_labels=sample_labels,
            max_length=64,
        )
        
        assert len(dataset) == 4
        assert dataset.max_length == 64
    
    def test_dataset_length(self, sample_dataframe, sample_labels):
        """Test dataset length."""
        dataset = MultiTaskDataset(sample_dataframe, sample_labels)
        assert len(dataset) == len(sample_dataframe)
    
    def test_getitem(self, sample_dataframe, sample_labels):
        """Test getting a single item."""
        dataset = MultiTaskDataset(sample_dataframe, sample_labels, max_length=64)
        
        item = dataset[0]
        
        # Check required keys
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'error_detection_labels' in item
        assert 'duplicate_detection_labels' in item
        assert 'outlier_detection_labels' in item
        
        # Check tensor types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        
        # Check shapes
        assert item['input_ids'].shape == (64,)
        assert item['attention_mask'].shape == (64,)
    
    def test_row_to_text(self, sample_dataframe, sample_labels):
        """Test row to text conversion."""
        dataset = MultiTaskDataset(sample_dataframe, sample_labels)
        
        row = sample_dataframe.iloc[0]
        text = dataset._row_to_text(row)
        
        assert 'name' in text
        assert 'Alice' in text
        assert 'age' in text
        assert '25' in text
    
    def test_default_tokenizer(self, sample_dataframe, sample_labels):
        """Test default tokenizer."""
        dataset = MultiTaskDataset(sample_dataframe, sample_labels, max_length=32)
        
        text = "Hello World"
        tokens = dataset._default_tokenizer(text, max_length=32)
        
        assert 'input_ids' in tokens
        assert 'attention_mask' in tokens
        assert len(tokens['input_ids']) == 32
        assert len(tokens['attention_mask']) == 32
    
    def test_collate_fn(self, sample_dataframe, sample_labels):
        """Test collate function for batching."""
        dataset = MultiTaskDataset(sample_dataframe, sample_labels, max_length=32)
        
        batch = [dataset[0], dataset[1], dataset[2]]
        collated = MultiTaskDataset.collate_fn(batch)
        
        # Check batch dimensions
        assert collated['input_ids'].shape == (3, 32)
        assert collated['attention_mask'].shape == (3, 32)
        assert collated['error_detection_labels'].shape == (3,)
    
    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['a', None, 'c', 'd'],
        })
        
        labels = {
            'error_detection': pd.Series([0, 0, 1, 0]),
        }
        
        dataset = MultiTaskDataset(df, labels)
        item = dataset[2]
        
        # Should handle NaN values
        assert item is not None
        text = dataset._row_to_text(df.iloc[2])
        assert '[MISSING]' in text
    
    def test_custom_tokenizer(self, sample_dataframe, sample_labels):
        """Test with custom tokenizer."""
        def custom_tokenizer(text, max_length):
            # Simple word-level tokenizer
            words = text.split()[:max_length]
            input_ids = [hash(w) % 1000 for w in words]
            
            # Pad
            if len(input_ids) < max_length:
                attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
                input_ids = input_ids + [0] * (max_length - len(input_ids))
            else:
                attention_mask = [1] * max_length
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        
        dataset = MultiTaskDataset(
            sample_dataframe,
            sample_labels,
            tokenizer=custom_tokenizer,
            max_length=32,
        )
        
        item = dataset[0]
        assert item['input_ids'].shape == (32,)
    
    def test_label_validation(self, sample_dataframe):
        """Test label validation."""
        # Mismatched label length should raise error
        invalid_labels = {
            'error_detection': pd.Series([0, 1]),  # Wrong length
        }
        
        with pytest.raises(AssertionError):
            MultiTaskDataset(sample_dataframe, invalid_labels)
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        empty_df = pd.DataFrame()
        labels = {}
        
        with pytest.raises(AssertionError):
            MultiTaskDataset(empty_df, labels)
    
    def test_multiple_label_types(self):
        """Test handling different label types."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
        })
        
        labels = {
            'task1': pd.Series([0, 1, 0, 1]),  # int
            'task2': pd.Series([0.5, 0.7, 0.3, 0.9]),  # float
            'task3': pd.Series(['a', 'b', 'a', 'c']),  # string
        }
        
        dataset = MultiTaskDataset(df, labels)
        item = dataset[0]
        
        # All labels should be converted to tensors
        assert isinstance(item['task1_labels'], torch.Tensor)
        assert isinstance(item['task2_labels'], torch.Tensor)
        assert isinstance(item['task3_labels'], torch.Tensor)


class TestDataLoader:
    """Test dataset with DataLoader."""
    
    def test_with_dataloader(self):
        """Test dataset integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20),
        })
        
        labels = {
            'error_detection': pd.Series(np.random.randint(0, 2, 10)),
        }
        
        dataset = MultiTaskDataset(df, labels, max_length=32)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=MultiTaskDataset.collate_fn,
        )
        
        # Test iteration
        for batch in dataloader:
            assert batch['input_ids'].shape[0] <= 4
            assert batch['input_ids'].shape[1] == 32
            assert 'error_detection_labels' in batch
            break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
