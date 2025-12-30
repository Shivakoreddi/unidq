"""
UNIDQ: Unified Data Quality
===========================

A unified transformer-based architecture for multi-task tabular data quality.

Handles 6 tasks simultaneously:
1. Error Detection - Identify erroneous cells
2. Data Repair - Correct erroneous values
3. Missing Value Imputation - Fill missing values
4. Label Noise Detection - Identify mislabeled samples
5. Label Classification - Classify samples
6. Data Valuation - Score data quality

Installation:
    pip install unidq

Quick Start:
    >>> from unidq import UNIDQ, MultiTaskDataset, UNIDQTrainer
    >>> 
    >>> # Create dataset
    >>> dataset = MultiTaskDataset(
    ...     dirty_features=X_dirty,
    ...     clean_features=X_clean,
    ...     error_mask=errors,
    ...     labels=y
    ... )
    >>> 
    >>> # Train model
    >>> model = UNIDQ(n_features=X.shape[1])
    >>> trainer = UNIDQTrainer(model)
    >>> trainer.fit(dataset, epochs=50)
    >>> 
    >>> # Predict
    >>> results = model.predict(X_new)

Paper: "UNIDQ: A Unified Transformer for Multi-Task Data Quality" (VLDB 2026)
GitHub: https://github.com/Shivakoreddi/unidq
"""

__version__ = "0.2.0"
__author__ = "Shiva Koreddi"

from .model import UNIDQ
from .dataset import MultiTaskDataset
from .trainer import UNIDQTrainer
from .evaluation import evaluate_all_tasks, evaluate_task
from .config import UNIDQConfig

__all__ = [
    "UNIDQ",
    "MultiTaskDataset",
    "UNIDQTrainer",
    "UNIDQConfig",
    "evaluate_all_tasks",
    "evaluate_task",
    "__version__",
]
