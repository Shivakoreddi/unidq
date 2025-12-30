# UNIDQ: Unified Data Quality

[![PyPI version](https://badge.fury.io/py/unidq.svg)](https://badge.fury.io/py/unidq)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-VLDB%202026-green.svg)](https://vldb.org/)

**UNIDQ** is a unified transformer-based architecture for multi-task tabular data quality. A single model handles 6 data quality tasks simultaneously, replacing the need for multiple specialized tools.

## 🎯 Six Tasks, One Model

| Task | Description | Metric |
|------|-------------|--------|
| **Error Detection** | Identify erroneous cells | F1: 0.894 |
| **Data Repair** | Correct erroneous values | R²: 0.539 |
| **Imputation** | Fill missing values | R²: 0.941 |
| **Label Noise Detection** | Identify mislabeled samples | F1: 0.856 |
| **Label Classification** | Classify samples | Acc: 0.922 |
| **Data Valuation** | Score data quality | ρ: 0.336 |

## 📦 Installation

```bash
pip install unidq
```

## 🚀 Quick Start

```python
from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer

# 1. Prepare your data
dataset = MultiTaskDataset(
    dirty_features=X_dirty,      # Corrupted data
    clean_features=X_clean,      # Ground truth (optional, for training)
    error_mask=errors,           # Binary error mask (optional)
    labels=y_noisy,              # Observed labels
    clean_labels=y_clean,        # True labels (optional)
    noise_mask=noise_mask,       # Label noise mask (optional)
)

# 2. Create and train model
model = UNIDQ(n_features=X_dirty.shape[1])
trainer = UNIDQTrainer(model)
trainer.fit(dataset, epochs=50)

# 3. Make predictions
results = model.predict(X_new)

print(results['error_mask'])      # Detected errors
print(results['repaired'])        # Repaired values
print(results['noise_mask'])      # Detected noisy labels
print(results['quality_scores'])  # Data quality scores
```

## 📊 Cross-Validation Example

```python
from unidq import UNIDQ, MultiTaskDataset
from unidq.trainer import cross_validate

# Load your dataset
dataset = MultiTaskDataset(
    dirty_features=X_dirty,
    clean_features=X_clean,
    error_mask=errors,
    labels=y_noisy,
    clean_labels=y_clean,
    noise_mask=noise_mask,
)

# Run 5-fold cross-validation
results = cross_validate(
    model_class=UNIDQ,
    dataset=dataset,
    n_features=X_dirty.shape[1],
    n_folds=5,
    epochs=50,
)

# Results contain mean ± std for all metrics
print(f"Error F1: {results['error_f1']['mean']:.3f} ± {results['error_f1']['std']:.3f}")
print(f"Noise F1: {results['noise_f1']['mean']:.3f} ± {results['noise_f1']['std']:.3f}")
```

## 🔧 Model Architecture

UNIDQ uses a shared transformer encoder with task-specific heads:

```
Input Features → [Value Embed + Z-Score Embed + Pos Embed]
                              ↓
                    Transformer Encoder (3 layers)
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        Cell Outputs      CLS Token      CLS Token
              ↓               ↓               ↓
    ┌─────────┼─────────┐     │     ┌─────────┼─────────┐
    ↓         ↓         ↓     ↓     ↓         ↓         ↓
  Error    Repair    Impute Label  Noise   Value
  Head     Head      Head   Head   Head    Head
```

**Model Size:** ~495K parameters (default configuration)

## 📈 Configuration

```python
from unidq import UNIDQ, UNIDQConfig

# Custom configuration
config = UNIDQConfig(
    d_model=256,      # Hidden dimension
    n_heads=8,        # Attention heads
    n_layers=6,       # Transformer layers
    dropout=0.1,
)

model = UNIDQ(n_features=14, config=config)
print(f"Parameters: {model.get_num_parameters():,}")
```

### Preset Configurations

```python
from unidq.config import UNIDQ_SMALL, UNIDQ_BASE, UNIDQ_LARGE

# UNIDQ_SMALL: 64d, 2 heads, 2 layers (~125K params)
# UNIDQ_BASE:  128d, 4 heads, 3 layers (~495K params) [default]
# UNIDQ_LARGE: 256d, 8 heads, 6 layers (~2M params)
```

## 📋 API Reference

### MultiTaskDataset

```python
dataset = MultiTaskDataset(
    dirty_features,    # np.ndarray (n_samples, n_features) - Required
    clean_features,    # np.ndarray - Ground truth features
    error_mask,        # np.ndarray - Binary error indicators
    missing_mask,      # np.ndarray - Binary missing indicators
    labels,            # np.ndarray - Observed labels
    clean_labels,      # np.ndarray - True labels
    noise_mask,        # np.ndarray - Binary noise indicators
)
```

### UNIDQ Model

```python
model = UNIDQ(
    n_features,        # int - Number of input features
    d_model=128,       # int - Hidden dimension
    n_heads=4,         # int - Attention heads
    n_layers=3,        # int - Transformer layers
    dropout=0.1,       # float - Dropout rate
)

# Forward pass
outputs = model(features, z_scores, labels)

# Prediction
results = model.predict(X_new, threshold=0.5)
```

### UNIDQTrainer

```python
trainer = UNIDQTrainer(model, device='cuda')

trainer.fit(
    train_dataset,
    val_dataset=None,
    epochs=50,
    batch_size=64,
    learning_rate=5e-4,
    patience=10,
)

metrics = trainer.evaluate(test_dataset)
```

## 🔬 Evaluation Metrics

```python
from unidq import evaluate_all_tasks
from unidq.evaluation import print_evaluation_report

metrics = evaluate_all_tasks(model, dataloader, device)
print_evaluation_report(metrics)
```

Output:
```
============================================================
UNIDQ Evaluation Report
============================================================

📌 ERROR DETECTION
   F1 Score:    0.8940
   ROC AUC:     0.9120
   Precision:   0.8650
   Recall:      0.9250

🔧 DATA REPAIR
   R² Score:    0.5390
   MAE:         0.1230

📥 IMPUTATION
   R² Score:    0.9410
   MAE:         0.0540

🏷️ LABEL NOISE DETECTION
   F1 Score:    0.8560
   ROC AUC:     0.8890

🎯 LABEL CLASSIFICATION
   Accuracy:    0.9220
   F1 Score:    0.9150

💰 DATA VALUATION
   Correlation: 0.3360
============================================================
```

## 📄 Citation

If you use UNIDQ in your research, please cite:

```bibtex
@inproceedings{koreddi2026unidq,
  title={UNIDQ: A Unified Transformer for Multi-Task Data Quality},
  author={Koreddi, Shiva},
  booktitle={Proceedings of the VLDB Endowment},
  year={2026}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 📋 Release Methodology

UNIDQ follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR** (e.g., 1.0.0 → 2.0.0): Breaking API changes
- **MINOR** (e.g., 0.1.0 → 0.2.0): New features, backwards compatible
- **PATCH** (e.g., 0.2.0 → 0.2.1): Bug fixes, backwards compatible

### Release Schedule

- **Patch releases**: As needed for critical bug fixes
- **Minor releases**: Monthly or when significant features are ready
- **Major releases**: When necessary for breaking changes

### PyTorch Compatibility

UNIDQ supports the latest two major PyTorch releases. We update within 30 days of new PyTorch releases to ensure compatibility.

Current support:
- PyTorch 1.9.0+
- PyTorch 2.0.0+
- PyTorch 2.1.0+

### How to Stay Updated

```bash
# Upgrade to latest version
pip install --upgrade unidq

# Check your version
python -c "import unidq; print(unidq.__version__)"
```

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

For governance details, see [GOVERNANCE.md](GOVERNANCE.md).

## 📧 Contact

- **Authors:** Shiva Koreddi, Sravani Sowrupilli
- **GitHub:** [@Shivakoreddi](https://github.com/Shivakoreddi)
- **Issues:** [GitHub Issues](https://github.com/Shivakoreddi/unidq/issues)
- **Email:** shivacse14@gmail.com, sravani.sowrupilli@gmail.com
