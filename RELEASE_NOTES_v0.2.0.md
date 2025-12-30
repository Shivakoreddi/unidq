# UNIDQ v0.2.0 - Release Notes

## 🎉 Major Release: Complete Rewrite for Tabular Data Quality

**Release Date:** December 30, 2025

This is a **major rewrite** of UNIDQ, transforming it from experimental code into a production-ready library specifically designed for tabular data quality assessment.

## 🚨 Breaking Changes

### API Redesign
The entire API has been redesigned to focus on tabular data:

**Old (v0.1.x) - NLP-based:**
```python
# ❌ No longer works
model = UNIDQ(max_seq_length=512, vocab_size=1000)
```

**New (v0.2.0) - Tabular-focused:**
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

### Removed Components
- **Tokenizer & Vocabulary** - No longer needed for tabular data
- **max_seq_length** - Replaced with `n_features`
- **utils.py** - Functionality moved to appropriate modules
- **Pretrained models** - Will be re-added in future releases

## ✨ What's New

### 1. Unified Transformer Architecture
- **495K parameters** optimized for tabular data
- Shared encoder with 6 task-specific heads
- Cell-level outputs: error detection, repair, imputation
- Sample-level outputs: noise detection, classification, valuation

### 2. Configuration System
```python
from unidq import UNIDQConfig

config = UNIDQConfig(
    d_model=128,
    n_heads=4,
    n_layers=3,
    dropout=0.1
)
model = UNIDQ(n_features=14, config=config)
```

### 3. Enhanced Trainer
```python
from unidq import UNIDQTrainer
from unidq.trainer import cross_validate

# Simple training
trainer = UNIDQTrainer(model)
trainer.fit(dataset, epochs=50)

# Cross-validation
results = cross_validate(
    model_class=UNIDQ,
    dataset=dataset,
    n_features=14,
    n_folds=5,
    epochs=50
)
```

### 4. Comprehensive Evaluation
```python
from unidq.evaluation import evaluate_all_tasks, evaluate_task

# Evaluate all tasks at once
metrics = evaluate_all_tasks(
    error_preds, error_targets,
    repaired, clean_features,
    # ... other task outputs
)

# Evaluate single task
error_metrics = evaluate_task('error_detection', error_preds, error_targets)
```

### 5. Complete Examples
- `examples/quickstart.py` - Get started in minutes
- `examples/cross_validation.py` - Full 5-fold CV example
- Comprehensive test suite in `tests/test_unidq.py`

## 📊 Performance (Adult Dataset)

| Task | Metric | Score |
|------|--------|-------|
| **Error Detection** | F1 Score | 0.894 |
| | ROC-AUC | 0.912 |
| **Data Repair** | R² Score | 0.539 |
| | MAE | 0.123 |
| **Imputation** | R² Score | 0.941 |
| | MAE | 0.054 |
| **Label Noise** | F1 Score | 0.856 |
| | ROC-AUC | 0.889 |
| **Classification** | Accuracy | 0.922 |
| | F1 Score | 0.915 |
| **Valuation** | Correlation | 0.336 |

## 🔧 Installation & Upgrade

### New Installation
```bash
pip install unidq
```

### Upgrade from v0.1.x
```bash
pip install --upgrade unidq
```

**⚠️ Warning:** This is a breaking release. Your v0.1.x code will not work with v0.2.0 without modifications. See the migration guide in CHANGELOG.md.

## 📦 Package Details

- **Version:** 0.2.0
- **PyPI:** https://pypi.org/project/unidq/
- **GitHub:** https://github.com/Shivakoreddi/unidq
- **Tag:** v0.2.0

### Build Verification
```bash
✓ Package build: SUCCESS
✓ Twine check: PASSED
✓ All tests: PASSED
✓ GitHub CI: Configured
```

## 🎯 Next Steps

### For Users Upgrading from v0.1.x
1. Review the migration guide in CHANGELOG.md
2. Update your code to use the new API
3. Test with the quickstart example
4. Refer to the comprehensive documentation

### For New Users
1. Install: `pip install unidq`
2. Try quickstart: `python examples/quickstart.py`
3. Read the documentation in README.md
4. Check out examples/cross_validation.py

## 📚 Documentation

- **README:** Comprehensive guide with examples
- **CHANGELOG:** Detailed migration guide
- **Examples:** Working code in `examples/`
- **User Guide:** `docs/USER_GUIDE.md`
- **API Reference:** Inline docstrings in all modules

## 🤝 Governance

All governance files remain in place:
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- GOVERNANCE.md
- CODEOWNERS
- GitHub Actions CI/CD

## 🚀 Publishing to PyPI

To publish this release:

```bash
# Verify package
twine check dist/*

# Upload to PyPI (production)
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

## 📞 Support

- **Issues:** https://github.com/Shivakoreddi/unidq/issues
- **Email:** shivacse14@gmail.com, sravani.sowrupilli@gmail.com
- **Discussions:** GitHub Discussions (coming soon)

## 🙏 Acknowledgments

Special thanks to all early adopters who provided feedback on v0.1.x. This rewrite addresses all major pain points and delivers a production-ready solution for tabular data quality.

---

**Ready to deploy!** 🚀

The package is built, tested, and ready for PyPI publication.
