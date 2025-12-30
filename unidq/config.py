"""
UNIDQ Configuration
===================

Configuration classes for UNIDQ model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class UNIDQConfig:
    """
    Configuration for UNIDQ model.
    
    Parameters
    ----------
    d_model : int, default=128
        Transformer hidden dimension
    n_heads : int, default=4
        Number of attention heads
    n_layers : int, default=3
        Number of transformer encoder layers
    dropout : float, default=0.1
        Dropout rate
    n_classes : int, default=2
        Number of classes for classification tasks
    task_weights : Dict[str, float], optional
        Weights for multi-task loss
        
    Example
    -------
    >>> config = UNIDQConfig(d_model=256, n_layers=6)
    >>> model = UNIDQ(n_features=14, config=config)
    """
    
    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    n_classes: int = 2
    
    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    
    # Task weights
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'error': 1.0,
        'repair': 0.5,
        'impute': 0.5,
        'label_clf': 0.3,
        'noise': 0.5,
        'value': 0.3,
    })
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.n_classes >= 2, "n_classes must be at least 2"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'n_classes': self.n_classes,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'task_weights': self.task_weights,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'UNIDQConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Preset configurations
UNIDQ_SMALL = UNIDQConfig(
    d_model=64,
    n_heads=2,
    n_layers=2,
    dropout=0.1,
)

UNIDQ_BASE = UNIDQConfig(
    d_model=128,
    n_heads=4,
    n_layers=3,
    dropout=0.1,
)

UNIDQ_LARGE = UNIDQConfig(
    d_model=256,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
)
