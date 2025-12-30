#!/usr/bin/env python3
"""
UNIDQ Tests
===========

Basic tests to verify library functionality.
"""

import numpy as np
import torch
import pytest


def test_imports():
    """Test that all modules can be imported."""
    from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer, UNIDQConfig
    from unidq import evaluate_all_tasks
    assert UNIDQ is not None
    assert MultiTaskDataset is not None
    assert UNIDQTrainer is not None
    assert UNIDQConfig is not None


def test_model_creation():
    """Test model creation with different configurations."""
    from unidq import UNIDQ, UNIDQConfig
    
    # Default configuration
    model = UNIDQ(n_features=10)
    assert model.n_features == 10
    assert model.d_model == 128
    assert model.n_heads == 4
    assert model.n_layers == 3
    
    # Custom configuration
    config = UNIDQConfig(d_model=64, n_heads=2, n_layers=2)
    model = UNIDQ(n_features=10, config=config)
    assert model.d_model == 64
    
    # Parameter count
    n_params = model.get_num_parameters()
    assert n_params > 0
    print(f"Model parameters: {n_params:,}")


def test_model_forward():
    """Test model forward pass."""
    from unidq import UNIDQ
    
    model = UNIDQ(n_features=10)
    
    # Create dummy inputs
    batch_size = 8
    features = torch.randn(batch_size, 10)
    z_scores = torch.abs(features)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    outputs = model(features, z_scores, labels)
    
    # Check output shapes
    assert outputs['error_logits'].shape == (batch_size, 10, 2)
    assert outputs['repair_pred'].shape == (batch_size, 10)
    assert outputs['impute_pred'].shape == (batch_size, 10)
    assert outputs['label_logits'].shape == (batch_size, 2)
    assert outputs['noise_logits'].shape == (batch_size, 2)
    assert outputs['value_pred'].shape == (batch_size,)


def test_model_predict():
    """Test model prediction."""
    from unidq import UNIDQ
    
    model = UNIDQ(n_features=10)
    
    # Create dummy data
    X = np.random.randn(20, 10).astype(np.float32)
    
    # Predict
    results = model.predict(X)
    
    # Check output keys
    assert 'error_mask' in results
    assert 'error_probs' in results
    assert 'repaired' in results
    assert 'imputed' in results
    assert 'label_pred' in results
    assert 'noise_mask' in results
    assert 'quality_scores' in results
    
    # Check shapes
    assert results['error_mask'].shape == (20, 10)
    assert results['repaired'].shape == (20, 10)
    assert results['label_pred'].shape == (20,)
    assert results['quality_scores'].shape == (20,)


def test_dataset_creation():
    """Test MultiTaskDataset creation."""
    from unidq import MultiTaskDataset
    
    n_samples = 100
    n_features = 10
    
    # Create dummy data
    X_dirty = np.random.randn(n_samples, n_features).astype(np.float32)
    X_clean = X_dirty.copy()
    error_mask = np.zeros((n_samples, n_features), dtype=np.float32)
    error_mask[:10, :3] = 1.0  # Some errors
    labels = np.random.randint(0, 2, n_samples)
    
    # Create dataset
    dataset = MultiTaskDataset(
        dirty_features=X_dirty,
        clean_features=X_clean,
        error_mask=error_mask,
        labels=labels,
    )
    
    assert len(dataset) == n_samples
    
    # Get a sample
    sample = dataset[0]
    assert 'dirty_features' in sample
    assert 'z_scores' in sample
    assert 'error_mask' in sample
    assert 'quality_score' in sample


def test_dataset_statistics():
    """Test dataset statistics computation."""
    from unidq import MultiTaskDataset
    
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    error_mask = np.zeros((n_samples, n_features), dtype=np.float32)
    error_mask[:10] = 1.0  # 10% error rate
    
    dataset = MultiTaskDataset(dirty_features=X, error_mask=error_mask)
    stats = dataset.get_statistics()
    
    assert stats['n_samples'] == n_samples
    assert stats['n_features'] == n_features
    assert 0.09 < stats['error_rate'] < 0.11  # ~10%


def test_trainer():
    """Test trainer with small dataset."""
    from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer
    
    n_samples = 50
    n_features = 5
    
    # Create dummy data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    error_mask = np.random.randint(0, 2, (n_samples, n_features)).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)
    noise_mask = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    dataset = MultiTaskDataset(
        dirty_features=X,
        clean_features=X,
        error_mask=error_mask,
        labels=labels,
        noise_mask=noise_mask,
    )
    
    # Create model and trainer
    model = UNIDQ(n_features=n_features, d_model=32, n_heads=2, n_layers=1)
    trainer = UNIDQTrainer(model)
    
    # Train for a few epochs
    history = trainer.fit(dataset, epochs=3, batch_size=16, verbose=False)
    
    assert 'train_loss' in history
    assert len(history['train_loss']) == 3


def test_evaluation():
    """Test evaluation functions."""
    from unidq import UNIDQ, MultiTaskDataset
    from unidq.evaluation import evaluate_all_tasks, evaluate_task
    from torch.utils.data import DataLoader
    
    n_samples = 50
    n_features = 5
    
    # Create dummy data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    error_mask = np.random.randint(0, 2, (n_samples, n_features)).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)
    noise_mask = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    dataset = MultiTaskDataset(
        dirty_features=X,
        clean_features=X,
        error_mask=error_mask,
        labels=labels,
        noise_mask=noise_mask,
    )
    
    dataloader = DataLoader(dataset, batch_size=16)
    
    model = UNIDQ(n_features=n_features, d_model=32, n_heads=2, n_layers=1)
    
    # Evaluate all tasks
    results = evaluate_all_tasks(model, dataloader)
    
    assert 'error_f1' in results
    assert 'repair_r2' in results
    assert 'impute_r2' in results
    assert 'noise_f1' in results
    assert 'label_accuracy' in results
    assert 'value_correlation' in results
    
    # Evaluate single task
    error_results = evaluate_task(model, dataloader, 'error')
    assert 'error_f1' in error_results


def test_save_load():
    """Test model save and load."""
    import tempfile
    from unidq import UNIDQ
    
    model = UNIDQ(n_features=10)
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model.save(f.name)
        
        # Load
        loaded_model = UNIDQ.load(f.name)
        
        assert loaded_model.n_features == model.n_features
        assert loaded_model.d_model == model.d_model


def test_config():
    """Test configuration class."""
    from unidq import UNIDQConfig
    from unidq.config import UNIDQ_SMALL, UNIDQ_BASE, UNIDQ_LARGE
    
    # Default config
    config = UNIDQConfig()
    assert config.d_model == 128
    assert config.n_heads == 4
    
    # Custom config
    config = UNIDQConfig(d_model=256, n_layers=6)
    assert config.d_model == 256
    assert config.n_layers == 6
    
    # Preset configs
    assert UNIDQ_SMALL.d_model == 64
    assert UNIDQ_BASE.d_model == 128
    assert UNIDQ_LARGE.d_model == 256
    
    # To/from dict
    config_dict = config.to_dict()
    config2 = UNIDQConfig.from_dict(config_dict)
    assert config2.d_model == config.d_model


if __name__ == '__main__':
    # Run tests
    test_imports()
    print("✓ Imports OK")
    
    test_model_creation()
    print("✓ Model creation OK")
    
    test_model_forward()
    print("✓ Model forward OK")
    
    test_model_predict()
    print("✓ Model predict OK")
    
    test_dataset_creation()
    print("✓ Dataset creation OK")
    
    test_dataset_statistics()
    print("✓ Dataset statistics OK")
    
    test_trainer()
    print("✓ Trainer OK")
    
    test_evaluation()
    print("✓ Evaluation OK")
    
    test_save_load()
    print("✓ Save/Load OK")
    
    test_config()
    print("✓ Config OK")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
