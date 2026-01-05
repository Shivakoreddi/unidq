"""
Unit tests for UNIDQ model
"""

import pytest
import torch
import numpy as np
from unidq.model import UNIDQ
from unidq.config import UNIDQConfig


class TestUNIDQConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = UNIDQConfig()
        
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 3
        assert config.dropout == 0.1
        assert config.n_classes == 2
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = UNIDQConfig(
            d_model=64,
            n_heads=2,
            n_layers=2,
            dropout=0.2,
        )
        
        assert config.d_model == 64
        assert config.n_heads == 2
        assert config.n_layers == 2
        assert config.dropout == 0.2
    
    def test_config_validation(self):
        """Test configuration validation."""
        # d_model must be divisible by n_heads
        with pytest.raises(AssertionError):
            UNIDQConfig(d_model=128, n_heads=5)


class TestUNIDQ:
    """Test UNIDQ model."""
    
    @pytest.fixture
    def n_features(self):
        """Number of features for testing."""
        return 10
    
    @pytest.fixture
    def batch_size(self):
        """Batch size for testing."""
        return 4
    
    @pytest.fixture
    def model(self, n_features):
        """Create a test model."""
        return UNIDQ(
            n_features=n_features,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            n_classes=2,
        )
    
    def test_model_initialization(self, model, n_features):
        """Test model initialization."""
        assert model.n_features == n_features
        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 2
        assert model.n_classes == 2
        
        # Check that model components exist
        assert hasattr(model, 'feature_embed')
        assert hasattr(model, 'z_embed')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'error_head')
        assert hasattr(model, 'repair_head')
        assert hasattr(model, 'impute_head')
        assert hasattr(model, 'label_head')
        assert hasattr(model, 'noise_head')
        assert hasattr(model, 'value_head')
    
    def test_forward_pass(self, model, n_features, batch_size):
        """Test forward pass with dummy input."""
        # Create dummy inputs
        features = torch.randn(batch_size, n_features)
        z_scores = torch.abs(torch.randn(batch_size, n_features))
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        outputs = model(features, z_scores, labels)
        
        # Check output keys
        assert 'error_logits' in outputs
        assert 'repair_pred' in outputs
        assert 'impute_pred' in outputs
        assert 'label_logits' in outputs
        assert 'noise_logits' in outputs
        assert 'value_pred' in outputs
    
    def test_output_shapes(self, model, n_features, batch_size):
        """Test output tensor shapes."""
        features = torch.randn(batch_size, n_features)
        z_scores = torch.abs(torch.randn(batch_size, n_features))
        labels = torch.randint(0, 2, (batch_size,))
        
        outputs = model(features, z_scores, labels)
        
        # Cell-level outputs: [batch, n_features, *]
        assert outputs['error_logits'].shape == (batch_size, n_features, 2)
        assert outputs['repair_pred'].shape == (batch_size, n_features)
        assert outputs['impute_pred'].shape == (batch_size, n_features)
        
        # Sample-level outputs: [batch, *]
        assert outputs['label_logits'].shape == (batch_size, 2)
        assert outputs['noise_logits'].shape == (batch_size, 2)
        assert outputs['value_pred'].shape == (batch_size,)
    
    def test_forward_without_labels(self, model, n_features, batch_size):
        """Test forward pass without labels."""
        features = torch.randn(batch_size, n_features)
        z_scores = torch.abs(torch.randn(batch_size, n_features))
        
        # Should work without labels
        outputs = model(features, z_scores, dirty_labels=None)
        
        assert 'error_logits' in outputs
        assert 'label_logits' in outputs
        # noise_logits should be zeros when labels are None
        assert outputs['noise_logits'].sum() == 0
    
    def test_predict_method(self, model, n_features):
        """Test predict method."""
        # Create test data
        features = torch.randn(8, n_features)
        
        model.eval()
        with torch.no_grad():
            predictions = model.predict(features)
        
        # Check prediction keys
        assert 'error_mask' in predictions
        assert 'repaired' in predictions
        assert 'imputed' in predictions
        assert 'label_pred' in predictions
        assert 'noise_mask' in predictions
        assert 'quality_scores' in predictions
        
        # Check shapes
        assert predictions['error_mask'].shape == (8, n_features)
        assert predictions['repaired'].shape == (8, n_features)
        assert predictions['label_pred'].shape == (8,)
    
    def test_save_and_load(self, model, tmp_path):
        """Test saving and loading model."""
        save_path = tmp_path / "test_model.pt"
        
        # Save model
        model.save(str(save_path))
        
        # Check file exists
        assert save_path.exists()
        
        # Load model
        loaded_model = UNIDQ.load(str(save_path))
        
        # Compare configurations
        assert loaded_model.n_features == model.n_features
        assert loaded_model.d_model == model.d_model
        assert loaded_model.n_heads == model.n_heads
        assert loaded_model.n_layers == model.n_layers
        
        # Compare outputs
        features = torch.randn(2, model.n_features)
        z_scores = torch.abs(torch.randn(2, model.n_features))
        labels = torch.randint(0, 2, (2,))
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            outputs1 = model(features, z_scores, labels)
            outputs2 = loaded_model(features, z_scores, labels)
        
        # Check all outputs match
        for key in outputs1.keys():
            assert torch.allclose(outputs1[key], outputs2[key], atol=1e-5)
    
    def test_model_with_config(self, n_features):
        """Test model initialization with config."""
        config = UNIDQConfig(
            d_model=64,
            n_heads=2,
            n_layers=2,
            dropout=0.1,
        )
        
        model = UNIDQ(n_features=n_features, config=config)
        
        assert model.d_model == 64
        assert model.n_heads == 2
        assert model.n_layers == 2
    
    def test_gradient_flow(self, model, n_features, batch_size):
        """Test that gradients flow through the model."""
        features = torch.randn(batch_size, n_features, requires_grad=True)
        z_scores = torch.abs(torch.randn(batch_size, n_features))
        labels = torch.randint(0, 2, (batch_size,))
        
        outputs = model(features, z_scores, labels)
        
        # Compute a dummy loss
        loss = outputs['error_logits'].mean()
        loss.backward()
        
        # Check that gradients exist on input
        assert features.grad is not None
        
        # Check that at least some model parameters have gradients
        has_grads = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        assert has_grads, "No model parameters have gradients"


class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single sample."""
        model = UNIDQ(n_features=5, d_model=32, n_heads=2, n_layers=1)
        
        features = torch.randn(1, 5)
        z_scores = torch.abs(torch.randn(1, 5))
        labels = torch.tensor([0])
        
        outputs = model(features, z_scores, labels)
        
        assert outputs['error_logits'].shape == (1, 5, 2)
        assert outputs['label_logits'].shape == (1, 2)
    
    def test_large_batch(self):
        """Test with large batch."""
        model = UNIDQ(n_features=10, d_model=64, n_heads=4, n_layers=2)
        
        batch_size = 128
        features = torch.randn(batch_size, 10)
        z_scores = torch.abs(torch.randn(batch_size, 10))
        labels = torch.randint(0, 2, (batch_size,))
        
        model.eval()
        with torch.no_grad():
            outputs = model(features, z_scores, labels)
        
        assert outputs['error_logits'].shape == (batch_size, 10, 2)
    
    def test_wide_dataset(self):
        """Test with many features."""
        n_features = 100
        model = UNIDQ(n_features=n_features, d_model=64, n_heads=4, n_layers=2)
        
        features = torch.randn(4, n_features)
        z_scores = torch.abs(torch.randn(4, n_features))
        labels = torch.randint(0, 2, (4,))
        
        model.eval()
        with torch.no_grad():
            outputs = model(features, z_scores, labels)
        
        assert outputs['error_logits'].shape == (4, n_features, 2)
        assert outputs['repair_pred'].shape == (4, n_features)


class TestModelDevice:
    """Test model on different devices."""
    
    def test_cpu(self):
        """Test model on CPU."""
        model = UNIDQ(n_features=10, d_model=64, n_heads=4, n_layers=2)
        model = model.to('cpu')
        
        features = torch.randn(4, 10)
        z_scores = torch.abs(torch.randn(4, 10))
        labels = torch.randint(0, 2, (4,))
        
        outputs = model(features, z_scores, labels)
        
        assert outputs['error_logits'].device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test model on CUDA."""
        model = UNIDQ(n_features=10, d_model=64, n_heads=4, n_layers=2)
        model = model.to('cuda')
        
        features = torch.randn(4, 10).cuda()
        z_scores = torch.abs(torch.randn(4, 10)).cuda()
        labels = torch.randint(0, 2, (4,)).cuda()
        
        outputs = model(features, z_scores, labels)
        
        assert outputs['error_logits'].device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

