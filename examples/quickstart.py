#!/usr/bin/env python3
"""
UNIDQ Quick Start Example
=========================

This example shows how to use UNIDQ for multi-task data quality.
"""

import numpy as np
from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer
from unidq.evaluation import print_evaluation_report

# =============================================================================
# 1. Generate Sample Data
# =============================================================================

np.random.seed(42)
n_samples = 1000
n_features = 10

# Clean data
X_clean = np.random.randn(n_samples, n_features).astype(np.float32)
y_clean = (X_clean[:, 0] + X_clean[:, 1] > 0).astype(np.int64)

# Inject errors (10% error rate)
X_dirty = X_clean.copy()
error_mask = np.zeros((n_samples, n_features), dtype=np.float32)
n_errors = int(n_samples * n_features * 0.1)
error_idx = np.random.choice(n_samples * n_features, n_errors, replace=False)
for idx in error_idx:
    row, col = idx // n_features, idx % n_features
    X_dirty[row, col] += np.random.randn() * 2  # Add noise
    error_mask[row, col] = 1.0

# Inject label noise (10% noise rate)
y_noisy = y_clean.copy()
noise_mask = np.zeros(n_samples, dtype=np.float32)
n_noisy = int(n_samples * 0.1)
noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
y_noisy[noisy_idx] = 1 - y_noisy[noisy_idx]  # Flip labels
noise_mask[noisy_idx] = 1.0

# Missing values mask (5% missing)
missing_mask = np.zeros((n_samples, n_features), dtype=np.float32)
n_missing = int(n_samples * n_features * 0.05)
missing_idx = np.random.choice(n_samples * n_features, n_missing, replace=False)
for idx in missing_idx:
    row, col = idx // n_features, idx % n_features
    if error_mask[row, col] == 0:  # Don't mark errors as missing
        missing_mask[row, col] = 1.0

print(f"Data shape: {X_dirty.shape}")
print(f"Error rate: {error_mask.mean()*100:.1f}%")
print(f"Noise rate: {noise_mask.mean()*100:.1f}%")
print(f"Missing rate: {missing_mask.mean()*100:.1f}%")

# =============================================================================
# 2. Create Dataset
# =============================================================================

dataset = MultiTaskDataset(
    dirty_features=X_dirty,
    clean_features=X_clean,
    error_mask=error_mask,
    missing_mask=missing_mask,
    labels=y_noisy,
    clean_labels=y_clean,
    noise_mask=noise_mask,
)

# Split into train/val
train_size = int(0.8 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

print(f"\nTrain samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# =============================================================================
# 3. Train Model
# =============================================================================

import torch

model = UNIDQ(n_features=n_features)
print(f"\nModel parameters: {model.get_num_parameters():,}")

trainer = UNIDQTrainer(model)
history = trainer.fit(
    train_dataset,
    val_dataset,
    epochs=30,
    batch_size=64,
    verbose=True,
)

# =============================================================================
# 4. Evaluate
# =============================================================================

print("\n" + "="*60)
print("Final Evaluation on Validation Set")
print("="*60)

metrics = trainer.evaluate(val_dataset)
print_evaluation_report(metrics)

# =============================================================================
# 5. Make Predictions
# =============================================================================

# Predict on new data
test_data = X_dirty[:10]
predictions = model.predict(test_data)

print("\n" + "="*60)
print("Sample Predictions (first 10 samples)")
print("="*60)

print("\nDetected Errors (per cell):")
print(predictions['error_mask'])

print("\nDetected Noisy Labels:")
print(predictions['noise_mask'])

print("\nQuality Scores:")
print(predictions['quality_scores'])

# =============================================================================
# 6. Save/Load Model
# =============================================================================

# Save
model.save('unidq_model.pt')
print("\nModel saved to unidq_model.pt")

# Load
loaded_model = UNIDQ.load('unidq_model.pt')
print("Model loaded successfully!")
