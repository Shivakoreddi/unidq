# UNIDQ: Unified Data Quality

[![PyPI version](https://badge.fury.io/py/unidq.svg)](https://pypi.org/project/unidq/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A unified transformer architecture for multi-task data quality assessment.**

---

## What is UNIDQ?

UNIDQ (Unified Data Quality) is a deep learning library that handles multiple data quality tasks with a single model. Rather than using separate tools for each task, UNIDQ provides one unified solution for all your data quality needs.

### The Problem

Real-world data is messy. You often deal with:
- Wrong or corrupted values
- Missing data
- Mislabeled samples
- Low-quality records

Fixing these issues typically requires multiple tools, each with different configurations and APIs.

### The Solution

UNIDQ handles **6 data quality tasks** with a single model:

| Task | What it does |
|------|--------------|
| **Error Detection** | Find wrong/dirty values in your data |
| **Data Repair** | Fix the detected errors |
| **Imputation** | Fill in missing values |
| **Label Noise Detection** | Find mislabeled samples |
| **Label Classification** | Predict labels for your data |
| **Data Valuation** | Score each sample's quality |

### Key Benefits

- 🎯 **Unified** - One model handles all six tasks
- ⚡ **Fast** - Processes 200K records in under 20 minutes
- 🔧 **Simple** - Clean, intuitive API
- 📈 **Accurate** - State-of-the-art performance on benchmark datasets
- 🪶 **Lightweight** - Efficient transformer architecture

---

## Installation

```bash
pip install unidq
```

**Requirements:** Python 3.8+, PyTorch 1.9+

---

## Quick Start

```python
from unidq import UNIDQ

# Initialize model
model = UNIDQ(n_features=10)

# Detect errors in your data
errors = model.detect_errors(X)

# Impute missing values
X_filled = model.impute(X)

# Find mislabeled samples
noisy = model.detect_label_noise(X, y)
```

---

## Usage Guide

### 1. Error Detection

Find erroneous values in your dataset.

```python
from unidq import UNIDQ

model = UNIDQ(n_features=X.shape[1])
model.fit(train_data)

# Detect errors
results = model.detect_errors(X_dirty)

print(f"Found {results['count']} errors")
print(f"Error rate: {results['error_rate']:.2%}")
```

**What you get:**
- `predictions`: Binary mask (1 = error, 0 = clean)
- `probabilities`: Confidence scores for each cell
- `count`: Total number of errors found
- `error_rate`: Percentage of erroneous cells

---

### 2. Data Repair

Automatically fix detected errors.

```python
# Repair errors
X_repaired = model.repair(X_dirty)

# Or detect and repair in one step
X_clean, report = model.detect_and_repair(X_dirty)

print(f"Repaired {report['repairs_made']} values")
```

**What you get:**
- Cleaned data with errors corrected
- Report showing what was changed

---

### 3. Missing Value Imputation

Fill in missing values intelligently.

```python
import numpy as np

# Data with missing values
X_missing = X.copy()
X_missing[0, 2] = np.nan
X_missing[5, 1] = np.nan

# Impute
X_filled = model.impute(X_missing)

print("Missing values filled!")
```

**What you get:**
- Complete data with no missing values
- Imputation uses learned patterns from your data

---

### 4. Label Noise Detection

Find samples that might be mislabeled.

```python
# Find noisy labels
results = model.detect_label_noise(X, y)

# Get suspicious samples
suspicious_indices = results['flagged_indices']
print(f"Found {len(suspicious_indices)} potentially mislabeled samples")

# Review them
for idx in suspicious_indices[:5]:
    print(f"Sample {idx}: label={y[idx]}, suggested={results['suggested_labels'][idx]}")
```

**What you get:**
- `noise_scores`: Probability each label is wrong
- `flagged_indices`: Samples likely mislabeled
- `suggested_labels`: What the correct label might be

---

### 5. Data Valuation

Score how useful each sample is for training.

```python
# Get quality scores
scores = model.valuate(X, y)

# Filter to high-quality samples
high_quality = scores > 0.7
X_clean = X[high_quality]
y_clean = y[high_quality]

print(f"Kept {high_quality.sum()}/{len(X)} samples")
```

**What you get:**
- Score between 0 and 1 for each sample
- Higher = better quality, more useful for training

---

### 6. Full Data Quality Assessment

Run everything at once.

```python
# Complete assessment
report = model.assess_quality(X, y)

print("=== Data Quality Report ===")
print(f"Errors: {report['errors']['count']}")
print(f"Missing: {report['missing']['count']}")
print(f"Noisy labels: {report['noise']['count']}")
print(f"Average quality: {report['quality']['mean']:.2f}")

# Get cleaned data
X_clean = report['cleaned_data']
y_clean = report['cleaned_labels']
```

---

## Training Your Own Model

### Basic Training

```python
from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer

# Prepare dataset
dataset = MultiTaskDataset(
    dirty_features=X_dirty,
    clean_features=X_clean,
    error_mask=errors,
    labels=y
)

# Create and train model
model = UNIDQ(n_features=X.shape[1])
trainer = UNIDQTrainer(model)
trainer.fit(dataset, epochs=50)

# Save for later
model.save('my_model.pt')
```

### Loading a Saved Model

```python
model = UNIDQ(n_features=10)
model.load('my_model.pt')

# Ready to use
results = model.detect_errors(new_data)
```

---

## Working with DataFrames

UNIDQ works seamlessly with Pandas.

```python
import pandas as pd
from unidq import UNIDQ

# Load your data
df = pd.read_csv('my_data.csv')

# Create model from DataFrame
model = UNIDQ.from_dataframe(df)

# Detect errors
errors = model.detect_errors(df)

# Get cleaned DataFrame
df_clean = model.clean(df)
df_clean.to_csv('cleaned_data.csv', index=False)
```

---

## Configuration Options

### Model Settings

```python
model = UNIDQ(
    n_features=20,        # Number of features in your data
    d_model=128,          # Model dimension (default: 128)
    n_layers=3,           # Number of transformer layers (default: 3)
    dropout=0.1           # Dropout rate (default: 0.1)
)
```

### Training Settings

```python
trainer = UNIDQTrainer(
    model,
    batch_size=64,        # Batch size (default: 64)
    learning_rate=1e-3,   # Learning rate (default: 1e-3)
    early_stopping=10     # Stop if no improvement for N epochs
)
```

---

## Example: Cleaning Customer Data

```python
import pandas as pd
from unidq import UNIDQ

# Load messy customer data
df = pd.read_csv('customers.csv')
print(f"Loaded {len(df)} records")

# Initialize UNIDQ
model = UNIDQ.from_dataframe(df)

# Run full assessment
report = model.assess_quality(df)

# Print findings
print("\n📊 Data Quality Report")
print(f"  • Errors found: {report['errors']['count']}")
print(f"  • Missing values: {report['missing']['count']}")
print(f"  • Suspicious labels: {report['noise']['count']}")
print(f"  • Overall quality: {report['quality']['mean']:.0%}")

# Save cleaned data
df_clean = report['cleaned_data']
df_clean.to_csv('customers_clean.csv', index=False)
print(f"\n✅ Saved cleaned data!")
```

---

## Performance

### Scalability

UNIDQ efficiently handles datasets of various sizes:

| Dataset Size | Training Time |
|--------------|---------------|
| 10,000 | ~2 min |
| 50,000 | ~11 min |
| 100,000 | ~9 min |
| 200,000 | ~18 min |

### Accuracy

UNIDQ achieves strong results across all tasks:

| Task | Performance |
|------|-------------|
| Error Detection | F1 = 0.89 |
| Imputation | R² = 0.94 |
| Label Noise Detection | F1 = 0.86 |
| Label Classification | Accuracy = 0.98 |

---

## FAQ

**Q: Do I need labeled data to use UNIDQ?**

For error detection and imputation, you can use UNIDQ in unsupervised mode. For best results on all tasks, providing some labeled examples helps.

**Q: How much data do I need?**

UNIDQ works with datasets as small as 1,000 records. Performance improves with more data.

**Q: Can I use UNIDQ with categorical data?**

Yes! UNIDQ automatically handles both numerical and categorical features.

**Q: How do I choose which tasks to run?**

Use `assess_quality()` to run everything, or call individual methods like `detect_errors()` or `impute()` for specific tasks.

**Q: Can I use a pre-trained model?**

Yes, you can save and load models using `model.save()` and `model.load()`.

---

## Citation

If you use UNIDQ in your research:

```bibtex
@inproceedings{unidq2026,
  title={UNIDQ: A Unified Transformer Architecture for Multi-Task Data Quality},
  author={Koreddi, Shiva and Sowrupilli, Sravani},
  booktitle={Proceedings of the VLDB Endowment},
  year={2026}
}
```

---

## Links

- **PyPI**: [pypi.org/project/unidq](https://pypi.org/project/unidq/)
- **GitHub**: [github.com/Shivakoreddi/unidq](https://github.com/Shivakoreddi/unidq)
- **Issues**: [Report bugs or request features](https://github.com/Shivakoreddi/unidq/issues)

---

## License

MIT License - free for personal and commercial use.

---

<p align="center">
  <b>One model. Six tasks. Clean data.</b>
</p>
