# UNIDQ Documentation

Welcome to the UNIDQ (Unified Transformer for Multi-Task Data Quality) documentation.

## Overview

UNIDQ is a deep learning framework built on transformer architecture for handling multiple data quality tasks in a unified manner. It provides a single model that can simultaneously:

- **Detect Errors**: Identify incorrect or inconsistent data entries
- **Impute Missing Values**: Fill in missing data intelligently
- **Match Schemas**: Align columns across different datasets
- **Detect Duplicates**: Find duplicate records
- **Detect Outliers**: Identify anomalous data points

## Installation

```bash
pip install unidq
```

Or install from source:

```bash
git clone https://github.com/yourusername/unidq.git
cd unidq
pip install -e .
```

## Quick Start

```python
import pandas as pd
from unidq import UNIDQ, UNIDQConfig, MultiTaskDataset
from torch.utils.data import DataLoader

# Load your data
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago'],
})

# Create task labels
task_labels = {
    'error_detection': pd.Series([0, 1, 0]),
    'duplicate_detection': pd.Series([0, 0, 1]),
}

# Create dataset
dataset = MultiTaskDataset(df, task_labels)

# Initialize model
config = UNIDQConfig()
model = UNIDQ(config)

# Train and evaluate
# ... (see examples for full training loop)
```

## Core Components

### 1. Model (`UNIDQ`)

The main transformer-based model for multi-task learning.

**Key Methods:**
- `forward()`: Run inference on input data
- `save_pretrained()`: Save model weights
- `from_pretrained()`: Load pre-trained weights

### 2. Configuration (`UNIDQConfig`)

Configure model architecture and hyperparameters.

**Key Parameters:**
- `d_model`: Embedding dimension (default: 256)
- `n_heads`: Number of attention heads (default: 8)
- `n_layers`: Number of transformer layers (default: 6)
- `dropout`: Dropout rate (default: 0.1)

### 3. Dataset (`MultiTaskDataset`)

PyTorch dataset for multi-task data quality.

**Key Methods:**
- `__getitem__()`: Get a single sample
- `collate_fn()`: Batch samples for DataLoader

### 4. Trainer (`UNIDQTrainer`)

Training utilities with multi-task loss handling.

**Key Methods:**
- `train()`: Train for multiple epochs
- `evaluate()`: Evaluate on validation set
- `train_epoch()`: Single epoch training

### 5. Evaluation

Comprehensive metrics for each task.

**Functions:**
- `evaluate_all_tasks()`: Evaluate all tasks at once
- `evaluate_error_detection()`: Error detection metrics
- `evaluate_imputation()`: Imputation metrics
- Task-specific evaluation functions

## Architecture

UNIDQ uses a transformer encoder architecture with task-specific output heads:

```
Input Data → Tokenization → Transformer Encoder → Task Heads
                                                   ├─ Error Detection
                                                   ├─ Imputation
                                                   ├─ Schema Matching
                                                   ├─ Duplicate Detection
                                                   └─ Outlier Detection
```

## Usage Examples

### Training a Model

```python
from unidq import UNIDQTrainer

trainer = UNIDQTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)

history = trainer.train(
    num_epochs=10,
    save_dir='./checkpoints',
)
```

### Making Predictions

```python
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    
# Get task-specific predictions
error_predictions = outputs['error_detection']
duplicate_predictions = outputs['duplicate_detection']
```

### Custom Tokenization

```python
def custom_tokenizer(text, max_length):
    # Your tokenization logic
    return {
        'input_ids': [...],
        'attention_mask': [...],
    }

dataset = MultiTaskDataset(
    df,
    task_labels,
    tokenizer=custom_tokenizer,
)
```

## Advanced Topics

### Multi-Task Learning

UNIDQ uses shared representations across tasks with task-specific output heads. Task weights can be customized:

```python
trainer = UNIDQTrainer(
    model=model,
    train_dataloader=train_loader,
    task_weights={
        'error_detection': 1.5,  # Higher weight
        'imputation': 1.0,
        'duplicate_detection': 0.8,  # Lower weight
    }
)
```

### Fine-tuning Pre-trained Models

```python
# Load pre-trained model
model = UNIDQ.from_pretrained('./pretrained_model')

# Fine-tune on your data
trainer = UNIDQTrainer(model, train_loader)
trainer.train(num_epochs=5)
```

### Custom Loss Functions

```python
trainer.loss_functions['custom_task'] = nn.BCEWithLogitsLoss()
```

## API Reference

See the full API documentation at: [Coming Soon]

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use UNIDQ in your research, please cite:

```bibtex
@software{unidq2024,
  title={UNIDQ: Unified Transformer for Multi-Task Data Quality},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/unidq}
}
```

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: https://github.com/yourusername/unidq/issues
- Documentation: https://unidq.readthedocs.io
- Email: your.email@example.com
