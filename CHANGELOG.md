# Changelog

All notable changes to UNIDQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Pre-trained model weights
- Additional data quality tasks
- Performance optimizations
- Documentation improvements

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
