# UNIDQ: Unified Data Quality

[![PyPI version](https://badge.fury.io/py/unidq.svg)](https://pypi.org/project/unidq/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://pepy.tech/badge/unidq)](https://pepy.tech/project/unidq)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Shivakoreddi%2Funidq-blue?logo=github)](https://github.com/Shivakoreddi/unidq)
[![GitHub Stars](https://img.shields.io/github/stars/Shivakoreddi/unidq?style=social)](https://github.com/Shivakoreddi/unidq/stargazers)

**A unified transformer architecture for multi-task data quality assessment.**

---

## 🎯 Overview

**UNIDQ** (Unified Data Quality) is a deep learning framework that addresses multiple data quality challenges with a single, efficient model. Unlike traditional approaches that require separate tools for each task (Raha for error detection, MICE for imputation, Cleanlab for label noise), UNIDQ handles **6 data quality tasks simultaneously** using a unified transformer architecture.

### Why UNIDQ?

| Traditional Approach | UNIDQ Approach |
|---------------------|----------------|
| Multiple separate tools | Single unified model |
| Tool-specific configurations | Configuration-free |
| No knowledge sharing between tasks | Multi-task learning with shared representations |
| High cumulative overhead | 495K parameters total |

---

## ✨ Features

UNIDQ addresses **6 data quality tasks** with a single model:

| Task | Description | Output |
|------|-------------|--------|
| ✅ **Error Detection** | Identify erroneous values in your data | Binary mask of errors |
| ✅ **Data Repair** | Suggest corrections for detected errors | Repaired values |
| ✅ **Missing Value Imputation** | Fill in missing values intelligently | Imputed values |
| ✅ **Label Noise Detection** | Find mislabeled samples | Noise probability scores |
| ✅ **Label Classification** | Predict labels for downstream tasks | Class predictions |
| ✅ **Data Valuation** | Score each sample's quality/usefulness | Quality scores [0-1] |

### Architecture Highlights

- **Three-Tier Attention**: Cell-level → Row-level → Column-level attention for comprehensive data understanding
- **Task-Specific LoRA Adapters**: Efficient fine-tuning with minimal parameters
- **Nash Multi-Task Learning**: Balanced optimization across all tasks
- **Lightweight Design**: Only 495K parameters

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install unidq
```

### From Source

```bash
git clone https://github.com/Shivakoreddi/unidq.git
cd unidq
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9
- NumPy >= 1.19
- scikit-learn >= 0.24
- pandas >= 1.2

---

## 🚀 Quick Start

### Basic Usage

```python
from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer

# Prepare your data
# X_dirty: DataFrame or array with potential errors
# X_clean: Ground truth clean data (for training)
# error_mask: Binary mask indicating errors (1 = error, 0 = clean)
# labels: Target labels for classification

# Create dataset
dataset = MultiTaskDataset(
    dirty_features=X_dirty,
    clean_features=X_clean,
    error_mask=error_mask,
    labels=labels
)

# Initialize model
model = UNIDQ(n_features=X_dirty.shape[1])

# Train
trainer = UNIDQTrainer(model)
trainer.fit(dataset, epochs=50)

# Get predictions
results = model.predict(X_new)
```

---

## 📖 Detailed Usage

### 1. Error Detection

Detect erroneous values in your dataset:

```python
from unidq import UNIDQ, ErrorDetector

# Load pre-trained model or train your own
model = UNIDQ(n_features=10)
model.load_pretrained('path/to/checkpoint.pt')

# Detect errors
error_predictions = model.detect_errors(X_dirty)

# Returns: dict with
#   - 'predictions': Binary array (1 = error)
#   - 'probabilities': Confidence scores
#   - 'error_indices': List of (row, col) tuples

print(f"Found {error_predictions['predictions'].sum()} errors")
print(f"Error locations: {error_predictions['error_indices'][:5]}")
```

### 2. Data Repair

Automatically repair detected errors:

```python
# Detect and repair in one step
repaired_data, repair_report = model.detect_and_repair(X_dirty)

# Or repair specific cells
repairs = model.repair(
    X_dirty, 
    error_mask=error_predictions['predictions']
)

print(f"Repaired {len(repair_report)} values")
print(f"Sample repairs: {repair_report[:3]}")
```

### 3. Missing Value Imputation

Handle missing values intelligently:

```python
import numpy as np

# Create data with missing values
X_missing = X_dirty.copy()
X_missing[np.isnan(X_missing)] = np.nan  # or use None for DataFrames

# Impute missing values
X_imputed = model.impute(X_missing)

# Get imputation confidence
imputed_values, confidence = model.impute(X_missing, return_confidence=True)

print(f"Imputed {np.isnan(X_missing).sum()} missing values")
print(f"Average confidence: {confidence.mean():.3f}")
```

### 4. Label Noise Detection

Identify potentially mislabeled samples:

```python
# Detect noisy labels
noise_scores = model.detect_label_noise(X, y)

# Returns: dict with
#   - 'noise_probabilities': P(label is wrong) for each sample
#   - 'predicted_clean_labels': What the label should be
#   - 'flagged_indices': Samples with noise_prob > threshold

# Find suspicious samples
threshold = 0.5
suspicious = noise_scores['noise_probabilities'] > threshold
print(f"Found {suspicious.sum()} potentially mislabeled samples")

# Review flagged samples
for idx in noise_scores['flagged_indices'][:5]:
    print(f"Sample {idx}: current={y[idx]}, suggested={noise_scores['predicted_clean_labels'][idx]}")
```

### 5. Data Valuation

Score the quality and usefulness of each sample:

```python
# Get quality scores for each sample
quality_scores = model.valuate(X, y)

# Returns: array of scores in [0, 1]
#   - 1.0 = high quality, useful sample
#   - 0.0 = low quality, potentially harmful sample

# Use for data selection
high_quality_mask = quality_scores > 0.7
X_clean = X[high_quality_mask]
y_clean = y[high_quality_mask]

print(f"Kept {high_quality_mask.sum()}/{len(X)} high-quality samples")
print(f"Quality distribution: min={quality_scores.min():.3f}, max={quality_scores.max():.3f}")
```

### 6. Full Pipeline (All Tasks)

Run all tasks in one call:

```python
# Comprehensive data quality assessment
results = model.assess_quality(
    X_dirty,
    labels=y,
    tasks=['error_detection', 'repair', 'imputation', 'noise_detection', 'valuation']
)

# Access results
print("=== Data Quality Report ===")
print(f"Errors detected: {results['error_detection']['count']}")
print(f"Values repaired: {results['repair']['count']}")
print(f"Missing imputed: {results['imputation']['count']}")
print(f"Noisy labels: {results['noise_detection']['count']}")
print(f"Avg quality score: {results['valuation']['mean']:.3f}")

# Get cleaned data
X_cleaned = results['cleaned_data']
y_cleaned = results['cleaned_labels']
```

---

## ⚙️ Advanced Configuration

### Custom Model Configuration

```python
from unidq import UNIDQ, UNIDQConfig

# Configure model architecture
config = UNIDQConfig(
    d_model=128,           # Embedding dimension
    n_heads=4,             # Attention heads
    n_layers=3,            # Transformer layers
    dropout=0.1,           # Dropout rate
    use_lora=True,         # Enable LoRA adapters
    lora_rank=8,           # LoRA rank
    task_weights={         # Custom task weights
        'error_detection': 1.0,
        'repair': 0.5,
        'imputation': 0.5,
        'noise_detection': 1.0,
        'classification': 0.3,
        'valuation': 0.3
    }
)

model = UNIDQ(n_features=20, config=config)
```

### Training Configuration

```python
from unidq import UNIDQTrainer, TrainingConfig

# Configure training
train_config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-3,
    max_epochs=100,
    early_stopping_patience=10,
    optimizer='adamw',
    scheduler='cosine',
    gradient_clip=1.0,
    validation_split=0.15
)

trainer = UNIDQTrainer(model, config=train_config)

# Train with callbacks
trainer.fit(
    dataset,
    callbacks=[
        EarlyStoppingCallback(patience=10),
        ModelCheckpointCallback(save_path='checkpoints/'),
        TensorBoardCallback(log_dir='logs/')
    ]
)
```

### Working with Pandas DataFrames

```python
import pandas as pd
from unidq import UNIDQ

# Load your data
df_dirty = pd.read_csv('dirty_data.csv')
df_clean = pd.read_csv('clean_data.csv')  # Optional, for training

# UNIDQ handles DataFrames directly
model = UNIDQ.from_dataframe(df_dirty)

# Or specify column types
model = UNIDQ.from_dataframe(
    df_dirty,
    numerical_columns=['age', 'salary', 'score'],
    categorical_columns=['city', 'department', 'status'],
    label_column='target'
)

# Detect errors
errors = model.detect_errors(df_dirty)

# Get cleaned DataFrame
df_cleaned = model.clean(df_dirty)
df_cleaned.to_csv('cleaned_data.csv', index=False)
```

### Loading Benchmark Datasets

```python
from unidq.datasets import load_benchmark

# Load a benchmark dataset
data = load_benchmark('beers')

print(f"Dirty data shape: {data['dirty'].shape}")
print(f"Clean data shape: {data['clean'].shape}")
print(f"Error rate: {data['error_mask'].mean():.2%}")

# Available datasets
from unidq.datasets import list_benchmarks
print(list_benchmarks())
# ['beers', 'flights', 'rayyan', 'hospital', 'tax', ...]
```

---

## 🔬 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `UNIDQ` | Main model class |
| `MultiTaskDataset` | Dataset wrapper for training |
| `UNIDQTrainer` | Training loop handler |
| `UNIDQConfig` | Model configuration |
| `TrainingConfig` | Training configuration |

### UNIDQ Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `detect_errors(X)` | Detect erroneous values | Dict with predictions, probabilities |
| `repair(X, error_mask)` | Repair detected errors | Repaired array |
| `impute(X)` | Impute missing values | Imputed array |
| `detect_label_noise(X, y)` | Find mislabeled samples | Dict with noise scores |
| `valuate(X, y)` | Score sample quality | Quality scores array |
| `assess_quality(X, y)` | Run all tasks | Comprehensive report dict |
| `predict(X)` | Get all predictions | Dict with all outputs |
| `fit(dataset)` | Train the model | self |
| `save(path)` | Save model checkpoint | None |
| `load(path)` | Load model checkpoint | self |

---

## 🧪 Examples

### Example 1: Cleaning a Messy CSV

```python
import pandas as pd
from unidq import UNIDQ

# Load messy data
df = pd.read_csv('messy_customer_data.csv')

# Initialize and run UNIDQ
model = UNIDQ.from_dataframe(df)
report = model.assess_quality(df)

# Print summary
print(f"Found {report['total_issues']} data quality issues:")
print(f"  - {report['error_detection']['count']} errors")
print(f"  - {report['imputation']['count']} missing values")
print(f"  - {report['noise_detection']['count']} suspicious labels")

# Save cleaned data
report['cleaned_data'].to_csv('clean_customer_data.csv', index=False)
```

### Example 2: Training on Custom Data

```python
from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Create datasets
train_dataset = MultiTaskDataset(
    dirty_features=X_train_dirty,
    clean_features=X_train_clean,
    error_mask=train_errors,
    labels=y_train
)

val_dataset = MultiTaskDataset(
    dirty_features=X_val_dirty,
    clean_features=X_val_clean,
    error_mask=val_errors,
    labels=y_val
)

# Train
model = UNIDQ(n_features=X.shape[1])
trainer = UNIDQTrainer(model)
history = trainer.fit(train_dataset, val_dataset=val_dataset, epochs=50)

# Plot training curves
trainer.plot_history(history)

# Save model
model.save('my_unidq_model.pt')
```

### Example 3: Integration with Scikit-learn

```python
from unidq import UNIDQTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create sklearn-compatible transformer
unidq_transformer = UNIDQTransformer(
    tasks=['error_detection', 'repair', 'imputation']
)

# Build pipeline
pipeline = Pipeline([
    ('data_quality', unidq_transformer),
    ('classifier', RandomForestClassifier())
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repo
git clone 
cd unidq

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 unidq/
black unidq/
```

---

## 📄 Citation

If you use UNIDQ in your research, please cite our paper:

```bibtex
@inproceedings{unidq2026,
  title={UNIDQ: A Unified Transformer Architecture for Multi-Task Data Quality},
  author={Koreddi, Shiva and Sowrupilli, Sravani},
  booktitle={Proceedings of the VLDB Endowment},
  year={2026},
  publisher={VLDB Endowment}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Inspired by research in data quality and multi-task learning


---

## 📧 Contact

- **Issues**: [GitHub Issues]
- **Email**: shivacse14@gmail.com

---

<p align="center">
  <b>Made with ❤️ for the data quality community</b>
</p>