# Changelog

All notable changes to UNIDQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Pre-trained model weights
- Additional data quality tasks (time-series, images)
- Integration with PyTorch Lightning
- Performance optimizations for large datasets
- Automated hyperparameter tuning

## [0.2.0] - 2025-12-30

### 🎉 Major Rewrite - Complete Architecture Redesign

This is a **major rewrite** of UNIDQ, transitioning from experimental code to a production-ready library specifically designed for **tabular data quality**.

### Added
- **New Architecture:** Unified transformer model (495K parameters) for all 6 tasks
- **Config System:** `UNIDQConfig` for centralized configuration management
- **Evaluation Module:** Comprehensive evaluation with `evaluate_all_tasks()` and `evaluate_task()`
- **Cross-Validation:** Built-in k-fold cross-validation support in `UNIDQTrainer`
- **Examples:** Complete working examples (`quickstart.py`, `cross_validation.py`)
- **Comprehensive Tests:** Full test suite in `test_unidq.py`
- **Better API:** Cleaner, more intuitive API matching PyTorch conventions

### Changed
- **BREAKING:** Complete API redesign for tabular data (replaces NLP-based approach)
- **Model Architecture:** From generic NLP model to specialized tabular transformer
- **Input Format:** Now uses numpy arrays for tabular features (removed tokenizer/vocab)
- **Dataset:** `MultiTaskDataset` specifically for tabular multi-task learning
- **Trainer:** Enhanced `UNIDQTrainer` with cross-validation and better progress tracking

### Removed
- **BREAKING:** Removed NLP-specific components (tokenizer, vocabulary, max_seq_length)
- Removed `utils.py` (functionality moved to appropriate modules)
- Removed pretrained models (will be re-added in future version)

### Fixed
- Architecture now correctly handles tabular data with cell-level and sample-level predictions
- Proper multi-task loss balancing
- Improved training stability and convergence

### Migration Guide

**Old API (0.1.x):**
```python
# ❌ No longer works
model = UNIDQ(max_seq_length=512, vocab_size=1000)
```

**New API (0.2.0):**
```python
# ✅ New approach
from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer

dataset = MultiTaskDataset(
    dirty_features=X_dirty,
    clean_features=X_clean,
    error_mask=errors,
    labels=y
)

model = UNIDQ(n_features=X.shape[1])
trainer = UNIDQTrainer(model)
trainer.fit(dataset, epochs=50)
```

### Performance
- Error Detection: F1 = 0.894, ROC-AUC = 0.912
- Data Repair: R² = 0.539
- Imputation: R² = 0.941
- Label Noise: F1 = 0.856
- Classification: Accuracy = 0.922
- Valuation: Correlation = 0.336

## [0.1.5] - 2024-12-29

### Fixed
- Documentation links to GitHub repository
- Badge URLs in README

## [0.1.4] - 2024-12-29

### Added
- Comprehensive user documentation and usage guide
- Detailed examples for all 6 data quality tasks
- FAQ section
- Performance benchmarks
- Complete API usage examples

### Documentation
- Full usage guide with code examples
- DataFrame integration examples
- Training and configuration guides
- Troubleshooting section

## [0.1.3] - 2024-12-29

### Added
- GitHub repository integration
- GitHub badges in README
- Repository links in PyPI metadata

### Changed
- Updated project URLs to point to GitHub repository

## [0.1.2] - 2024-12-28

### Changed
- Updated author information
- Updated citation with author names

## [0.1.1] - 2024-12-28

### Changed
- Updated author metadata

## [0.1.0] - 2024-12-28

### Added
- Initial release of UNIDQ
- Core transformer model implementation (`UNIDQ`)
- Multi-task dataset class (`MultiTaskDataset`)
- Training utilities (`UNIDQTrainer`)
- Evaluation metrics for all tasks
- Support for 5 data quality tasks:
  - Error detection
  - Data imputation
  - Schema matching
  - Duplicate detection
  - Outlier detection
- Comprehensive test suite
- Example scripts and tutorial notebook
- Documentation and API reference
- MIT License
- PyPI package configuration

### Features
- Transformer-based architecture
- Multi-task learning with shared encoder
- Task-specific output heads
- Flexible configuration system
- Save/load pre-trained models
- Custom tokenizer support
- Batch processing support
- GPU acceleration support

### Documentation
- README with quick start guide
- API documentation
- Tutorial Jupyter notebook
- Example scripts
- Test coverage

### Dependencies
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- scikit-learn >= 0.24.0
- tqdm >= 4.60.0

## Version History

- **0.1.0** (2024-12-28): Initial release

---

[Unreleased]: https://github.com/yourusername/unidq/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/unidq/releases/tag/v0.1.0
