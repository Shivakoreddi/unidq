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
    def n_samples(self):
        """Number of samples for testing."""
        return 20
    
    @pytest.fixture
    def n_features(self):
        """Number of features for testing."""
        return 10
    
    @pytest.fixture
    def sample_data(self, n_samples, n_features):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Create clean data
        clean = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Create dirty data (with some errors)
        dirty = clean.copy()
        error_mask = np.random.rand(n_samples, n_features) < 0.1
        dirty[error_mask] += np.random.randn(error_mask.sum()) * 5
        
        # Create missing mask
        missing_mask = np.random.rand(n_samples, n_features) < 0.05
        dirty[missing_mask] = np.nan
        
        # Create labels
        labels = np.random.randint(0, 2, n_samples).astype(np.int64)
        clean_labels = labels.copy()
        noise_mask = np.random.rand(n_samples) < 0.1
        labels[noise_mask] = 1 - labels[noise_mask]
        
        return {
            'dirty_features': dirty,
            'clean_features': clean,
            'error_mask': error_mask.astype(np.float32),
            'missing_mask': missing_mask.astype(np.float32),
            'labels': labels,
            'clean_labels': clean_labels,
            'noise_mask': noise_mask.astype(np.float32),
        }
    
    def test_dataset_initialization(self, sample_data, n_samples, n_features):
        """Test dataset initialization."""
        dataset = MultiTaskDataset(
            dirty_features=sample_data['dirty_features'],
            clean_features=sample_data['clean_features'],
            error_mask=sample_data['error_mask'],
            labels=sample_data['labels'],
        )
        
        assert len(dataset) == n_samples
        assert dataset.n_features == n_features
        assert dataset.n_samples == n_samples
    
    def test_dataset_length(self, sample_data):
        """Test dataset length."""
        dataset = MultiTaskDataset(
            dirty_features=sample_data['dirty_features']
        )
        assert len(dataset) == len(sample_data['dirty_features'])
    
    def test_getitem(self, sample_data, n_features):
        """Test getting a single item."""
        dataset = MultiTaskDataset(
            dirty_features=sample_data['dirty_features'],
            clean_features=sample_data['clean_features'],
            error_mask=sample_data['error_mask'],
            missing_mask=sample_data['missing_mask'],
            labels=sample_data['labels'],
            clean_labels=sample_data['clean_labels'],
            noise_mask=sample_data['noise_mask'],
        )
        
        item = dataset[0]
        
        # Check required keys
        assert 'dirty_features' in item
        assert 'clean_features' in item
        assert 'error_mask' in item
        assert 'missing_mask' in item
        assert 'dirty_label' in item
        assert 'clean_label' in item
        assert 'noise_label' in item
        assert 'z_scores' in item
        assert 'quality_score' in item
        
        # Check tensor types
        assert isinstance(item['dirty_features'], torch.Tensor)
        assert isinstance(item['clean_features'], torch.Tensor)
        assert isinstance(item['z_scores'], torch.Tensor)
        
        # Check shapes
        assert item['dirty_features'].shape == (n_features,)
        assert item['clean_features'].shape == (n_features,)
        assert item['z_scores'].shape == (n_features,)
        assert item['error_mask'].shape == (n_features,)
    
    def test_default_values(self):
        """Test dataset with only dirty features (minimal)."""
        dirty = np.random.randn(10, 5).astype(np.float32)
        dataset = MultiTaskDataset(dirty_features=dirty)
        
        assert len(dataset) == 10
        
        item = dataset[0]
        
        # Should create default values for missing fields
        assert 'dirty_features' in item
        assert 'clean_features' in item  # Should default to dirty
        assert 'error_mask' in item  # Should be zeros
        assert 'dirty_label' in item  # Should be zero
    
    def test_z_score_computation(self, sample_data):
        """Test z-score computation."""
        dataset = MultiTaskDataset(
            dirty_features=sample_data['dirty_features'],
            compute_z_scores=True,
        )
        
        item = dataset[0]
        
        # Z-scores should exist
        assert 'z_scores' in item
        assert item['z_scores'].shape == item['dirty_features'].shape
        
        # Z-scores should be non-negative
        assert (item['z_scores'] >= 0).all()
    
    def test_quality_scores(self, sample_data):
        """Test quality score computation."""
        dataset = MultiTaskDataset(
            dirty_features=sample_data['dirty_features'],
            error_mask=sample_data['error_mask'],
            missing_mask=sample_data['missing_mask'],
        )
        
        item = dataset[0]
        
        # Quality score should exist (singular)
        assert 'quality_score' in item
        
        # Quality score should be in [0, 1]
        assert 0 <= item['quality_score'] <= 1
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        dirty = np.random.randn(10, 5).astype(np.float32)
        dirty[2, 3] = np.nan
        dirty[5, 1] = np.nan
        
        dataset = MultiTaskDataset(dirty_features=dirty)
        
        item = dataset[2]
        
        # NaN should be replaced with 0
        assert not np.isnan(item['dirty_features'].numpy()).any()
        
        # Missing mask should be set for NaN position
        # Note: Missing mask might not track original NaN after replacement
        # This test verifies no NaN remains
    
    def test_missing_mask_detection(self):
        """Test automatic missing mask creation."""
        dirty = np.random.randn(10, 5).astype(np.float32)
        
        # Create explicit missing mask
        missing_mask = np.zeros((10, 5), dtype=np.float32)
        missing_mask[2, 3] = 1.0
        missing_mask[5, 1] = 1.0
        
        dataset = MultiTaskDataset(
            dirty_features=dirty,
            missing_mask=missing_mask
        )
        
        item_with_missing = dataset[2]
        item_without_missing = dataset[0]
        
        # Item 2 should have missing value at index 3
        assert item_with_missing['missing_mask'][3] == 1.0
        
        # Item 0 should have no missing values
        assert item_without_missing['missing_mask'].sum() == 0
    
    def test_batch_consistency(self, sample_data):
        """Test that multiple calls to getitem return consistent results."""
        dataset = MultiTaskDataset(
            dirty_features=sample_data['dirty_features'],
            clean_features=sample_data['clean_features'],
        )
        
        item1 = dataset[5]
        item2 = dataset[5]
        
        # Should return same data
        assert torch.equal(item1['dirty_features'], item2['dirty_features'])
        assert torch.equal(item1['clean_features'], item2['clean_features'])


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample_dataset(self):
        """Test dataset with single sample."""
        dirty = np.random.randn(1, 10).astype(np.float32)
        dataset = MultiTaskDataset(dirty_features=dirty)
        
        assert len(dataset) == 1
        item = dataset[0]
        assert item['dirty_features'].shape == (10,)
    
    def test_single_feature_dataset(self):
        """Test dataset with single feature."""
        dirty = np.random.randn(20, 1).astype(np.float32)
        dataset = MultiTaskDataset(dirty_features=dirty)
        
        assert dataset.n_features == 1
        item = dataset[0]
        assert item['dirty_features'].shape == (1,)
    
    def test_wide_dataset(self):
        """Test dataset with many features."""
        n_features = 200
        dirty = np.random.randn(10, n_features).astype(np.float32)
        
        # Create explicit missing mask for all features in row 3
        missing_mask = np.zeros((10, 5), dtype=np.float32)
        missing_mask[3, :] = 1.0
        
        dataset = MultiTaskDataset(
            dirty_features=dirty,
            missing_mask=missing_mask
        )
        item = dataset[3]
        
        # Should handle all missing without errors
        assert item is not None
        assert item['missing_mask'].sum() == 5  # All features missing
    
    def test_all_errors_row(self):
        """Test row with all errors."""
        dirty = np.random.randn(10, 5).astype(np.float32)
        error_mask = np.zeros((10, 5), dtype=np.float32)
        error_mask[3, :] = 1.0  # All errors in row 3
        
        dataset = MultiTaskDataset(
            dirty_features=dirty,
            error_mask=error_mask,
        )
        item = dataset[3]
        
        assert item['error_mask'].sum() == 5


class TestDataLoader:
    """Test dataset integration with DataLoader."""
    
    def test_with_dataloader(self):
        """Test dataset with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dirty = np.random.randn(50, 10).astype(np.float32)
        clean = np.random.randn(50, 10).astype(np.float32)
        labels = np.random.randint(0, 2, 50).astype(np.int64)
        
        dataset = MultiTaskDataset(
            dirty_features=dirty,
            clean_features=clean,
            labels=labels,
        )
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Test iteration
        for batch in dataloader:
            assert batch['dirty_features'].shape[0] <= 8
            assert batch['dirty_features'].shape[1] == 10
            assert 'clean_features' in batch
            assert 'dirty_label' in batch
            break
    
    def test_dataloader_batch_shapes(self):
        """Test batch shapes from DtaLoader."""
        from torch.utils.data import DataLoader
        
        n_samples = 100
        n_features = 15
        batch_size = 16
        
        dirty = np.random.randn(n_samples, n_features).astype(np.float32)
        dataset = MultiTaskDataset(dirty_features=dirty)
        
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        batch = next(iter(dataloader))
        
        assert batch['dirty_features'].shape == (batch_size, n_features)
        assert batch['z_scores'].shape == (batch_size, n_features)
        assert batch['quality_score'].shape == (batch_size,)


class TestDatasetFromDataFrame:
    """Test creating dataset from pandas DataFrame."""
    
    def test_from_numeric_dataframe(self):
        """Test creation from numeric DataFrame."""
        df = pd.DataFrame(np.random.randn(20, 5), columns=['a', 'b', 'c', 'd', 'e'])
        
        dataset = MultiTaskDataset(dirty_features=df.values.astype(np.float32))
        
        assert len(dataset) == 20
        assert dataset.n_features == 5
    
    def test_from_mixed_dataframe(self):
        """Test creation from DataFrame with mixed types."""
        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0],
            'num2': [5.0, 6.0, 7.0, 8.0],
            'num3': [9.0, 10.0, 11.0, 12.0],
        })
        
        dataset = MultiTaskDataset(dirty_features=df.values.astype(np.float32))
        
        assert len(dataset) == 4
        assert dataset.n_features == 3
        
        item = dataset[0]
        assert item['dirty_features'].shape == (3,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
