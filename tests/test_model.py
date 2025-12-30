"""
Unit tests for UNIDQ model
"""

import pytest
import torch
from unidq.model import UNIDQ, UNIDQConfig


class TestUNIDQConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = UNIDQConfig()
        
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 1024
        assert config.dropout == 0.1
        assert config.max_seq_length == 512
        assert config.vocab_size == 10000
        assert config.num_tasks == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = UNIDQConfig(
            d_model=128,
            n_heads=4,
            n_layers=3,
        )
        
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 3


class TestUNIDQ:
    """Test UNIDQ model."""
    
    @pytest.fixture
    def config(self):
        """Create a small test configuration."""
        return UNIDQConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_length=32,
            vocab_size=100,
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return UNIDQ(config)
    
    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert model.config == config
        assert isinstance(model.token_embedding, torch.nn.Embedding)
        assert isinstance(model.position_embedding, torch.nn.Embedding)
        assert len(model.task_heads) == 5
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        outputs = model(input_ids, attention_mask)
        
        assert len(outputs) == 5
        assert 'error_detection' in outputs
        assert 'imputation' in outputs
        assert 'schema_matching' in outputs
        assert 'duplicate_detection' in outputs
        assert 'outlier_detection' in outputs
    
    def test_single_task_forward(self, model):
        """Test forward pass for single task."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        
        outputs = model(input_ids, task='error_detection')
        
        assert len(outputs) == 1
        assert 'error_detection' in outputs
    
    def test_output_shapes(self, model, config):
        """Test output tensor shapes."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        outputs = model(input_ids)
        
        # Error detection: [batch, seq, 2]
        assert outputs['error_detection'].shape == (batch_size, seq_length, 2)
        
        # Imputation: [batch, seq, vocab_size]
        assert outputs['imputation'].shape == (batch_size, seq_length, config.vocab_size)
        
        # Schema matching: [batch, seq, d_model]
        assert outputs['schema_matching'].shape == (batch_size, seq_length, config.d_model)
    
    def test_invalid_task(self, model):
        """Test error handling for invalid task."""
        input_ids = torch.randint(0, 100, (2, 16))
        
        with pytest.raises(ValueError):
            model(input_ids, task='invalid_task')
    
    def test_save_and_load(self, model, tmp_path):
        """Test saving and loading model."""
        save_dir = tmp_path / "test_model"
        
        # Save model
        model.save_pretrained(str(save_dir))
        
        # Check files exist
        assert (save_dir / "pytorch_model.bin").exists()
        assert (save_dir / "config.json").exists()
        
        # Load model
        loaded_model = UNIDQ.from_pretrained(str(save_dir))
        
        # Compare configurations
        assert loaded_model.config.d_model == model.config.d_model
        assert loaded_model.config.n_heads == model.config.n_heads
        
        # Compare outputs
        input_ids = torch.randint(0, 100, (2, 16))
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            outputs1 = model(input_ids)
            outputs2 = loaded_model(input_ids)
        
        for task in outputs1.keys():
            assert torch.allclose(outputs1[task], outputs2[task], atol=1e-5)


class TestModelComponents:
    """Test individual model components."""
    
    def test_embedding_layer(self):
        """Test embedding layers."""
        config = UNIDQConfig(d_model=64, vocab_size=100, max_seq_length=32)
        model = UNIDQ(config)
        
        input_ids = torch.randint(0, 100, (2, 16))
        token_embeds = model.token_embedding(input_ids)
        
        assert token_embeds.shape == (2, 16, 64)
    
    def test_task_heads(self):
        """Test task-specific heads."""
        config = UNIDQConfig(d_model=64, vocab_size=100)
        model = UNIDQ(config)
        
        hidden_states = torch.randn(2, 16, 64)
        
        # Error detection head
        error_output = model.task_heads['error_detection'](hidden_states)
        assert error_output.shape == (2, 16, 2)
        
        # Imputation head
        imputation_output = model.task_heads['imputation'](hidden_states)
        assert imputation_output.shape == (2, 16, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

