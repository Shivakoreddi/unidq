"""
UNIDQ Model
===========

Unified transformer-based model for multi-task tabular data quality.

Architecture:
- Shared transformer encoder (495K parameters)
- 6 task-specific heads
- Cell-level and sample-level outputs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union, Tuple
from .config import UNIDQConfig


class UNIDQ(nn.Module):
    """
    Unified Data Quality Model.
    
    A transformer-based architecture that handles 6 data quality tasks:
    1. Error Detection (cell-level)
    2. Data Repair (cell-level)
    3. Missing Value Imputation (cell-level)
    4. Label Noise Detection (sample-level)
    5. Label Classification (sample-level)
    6. Data Valuation (sample-level)
    
    Parameters
    ----------
    n_features : int
        Number of input features (columns)
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
        
    Example
    -------
    >>> model = UNIDQ(n_features=14)
    >>> outputs = model(features, z_scores, labels)
    >>> print(outputs.keys())
    dict_keys(['error_logits', 'repair_pred', 'impute_pred', 
               'label_logits', 'noise_logits', 'value_pred'])
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        n_classes: int = 2,
        config: Optional[UNIDQConfig] = None,
    ):
        super().__init__()
        
        # Use config if provided
        if config is not None:
            d_model = config.d_model
            n_heads = config.n_heads
            n_layers = config.n_layers
            dropout = config.dropout
            n_classes = config.n_classes
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # =====================================================================
        # Input Embeddings
        # =====================================================================
        
        # Feature value embedding
        self.feature_embed = nn.Linear(1, d_model)
        
        # Z-score embedding (captures statistical anomalies)
        self.z_embed = nn.Linear(1, d_model)
        
        # Positional embedding for column positions
        self.pos_embed = nn.Embedding(max(n_features + 1, 512), d_model)
        
        # CLS token for sample-level tasks
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # =====================================================================
        # Transformer Encoder
        # =====================================================================
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # =====================================================================
        # Task-Specific Heads
        # =====================================================================
        
        # Task 1: Error Detection (cell-level, binary classification)
        self.error_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
        # Task 2: Data Repair (cell-level, regression)
        self.repair_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Task 3: Missing Value Imputation (cell-level, regression)
        self.impute_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Task 4: Label Classification (sample-level)
        self.label_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        
        # Task 5: Label Noise Detection (sample-level)
        # Takes label probabilities + label one-hot + disagreement signal
        self.noise_head = nn.Sequential(
            nn.Linear(n_classes * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Task 6: Data Valuation (sample-level, regression 0-1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        features: torch.Tensor,
        z_scores: torch.Tensor,
        dirty_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all 6 tasks.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features, shape (batch_size, n_features)
        z_scores : torch.Tensor
            Z-scores for each feature, shape (batch_size, n_features)
        dirty_labels : torch.Tensor, optional
            Observed (potentially noisy) labels, shape (batch_size,)
            Required for noise detection task
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing outputs for all tasks:
            - 'error_logits': (batch, n_features, 2) - Error detection logits
            - 'repair_pred': (batch, n_features) - Repaired values
            - 'impute_pred': (batch, n_features) - Imputed values
            - 'label_logits': (batch, n_classes) - Label classification logits
            - 'noise_logits': (batch, 2) - Noise detection logits
            - 'value_pred': (batch,) - Data quality scores
        """
        batch_size, num_features = features.shape
        device = features.device
        
        # =====================================================================
        # Input Encoding
        # =====================================================================
        
        # Embed feature values: (batch, n_features) -> (batch, n_features, d_model)
        feat_emb = self.feature_embed(features.unsqueeze(-1))
        
        # Embed z-scores: captures statistical anomalies
        z_emb = self.z_embed(z_scores.unsqueeze(-1))
        
        # Combine embeddings
        x = feat_emb + z_emb
        
        # Add CLS token: (batch, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + n_features, d_model)
        
        # Add positional embeddings
        seq_len = min(num_features + 1, 512)
        positions = torch.arange(seq_len, device=device)
        x = self.norm(x + self.pos_embed(positions[:x.shape[1]]))
        
        # =====================================================================
        # Transformer Encoding
        # =====================================================================
        
        x = self.encoder(x)
        
        # Split outputs
        cls_out = x[:, 0]       # (batch, d_model) - for sample-level tasks
        feat_out = x[:, 1:]    # (batch, n_features, d_model) - for cell-level tasks
        
        # =====================================================================
        # Task Heads
        # =====================================================================
        
        outputs = {}
        
        # Task 1: Error Detection
        outputs['error_logits'] = self.error_head(feat_out)  # (batch, n_features, 2)
        
        # Task 2: Data Repair
        outputs['repair_pred'] = self.repair_head(feat_out).squeeze(-1)  # (batch, n_features)
        
        # Task 3: Missing Value Imputation
        outputs['impute_pred'] = self.impute_head(feat_out).squeeze(-1)  # (batch, n_features)
        
        # Task 4: Label Classification
        label_logits = self.label_head(cls_out)  # (batch, n_classes)
        outputs['label_logits'] = label_logits
        label_probs = torch.softmax(label_logits, dim=-1)
        
        # Task 5: Label Noise Detection
        if dirty_labels is not None:
            dirty_labels = dirty_labels.long().to(device)
            
            # Create one-hot encoding of observed labels
            label_onehot = torch.zeros(batch_size, self.n_classes, device=device)
            label_onehot.scatter_(1, dirty_labels.unsqueeze(1), 1.0)
            
            # Compute disagreement signal: probability of opposite class
            opposite_idx = 1 - dirty_labels  # For binary classification
            opposite_prob = label_probs.gather(1, opposite_idx.unsqueeze(-1))
            
            # Concatenate features for noise detection
            noise_features = torch.cat([label_probs, label_onehot, opposite_prob], dim=-1)
            outputs['noise_logits'] = self.noise_head(noise_features)  # (batch, 2)
        else:
            outputs['noise_logits'] = torch.zeros(batch_size, 2, device=device)
        
        # Task 6: Data Valuation
        outputs['value_pred'] = self.value_head(cls_out).squeeze(-1)  # (batch,)
        
        return outputs
    
    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor],
        z_scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        features : array-like, shape (n_samples, n_features)
            Input features (can be dirty)
        z_scores : array-like, optional
            Pre-computed z-scores. If None, computed automatically.
        labels : array-like, optional
            Observed labels for noise detection
        threshold : float, default=0.5
            Classification threshold for binary predictions
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing predictions:
            - 'error_mask': Binary error predictions
            - 'error_probs': Error probabilities
            - 'repaired': Repaired feature values
            - 'imputed': Imputed feature values
            - 'label_pred': Predicted labels
            - 'label_probs': Label probabilities
            - 'noise_mask': Binary noise predictions
            - 'noise_probs': Noise probabilities
            - 'quality_scores': Data quality scores
        """
        self.eval()
        
        # Convert to tensors
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Compute z-scores if not provided
        if z_scores is None:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + 1e-8
            z_scores = torch.abs((features - mean) / std)
        elif isinstance(z_scores, np.ndarray):
            z_scores = torch.FloatTensor(z_scores)
        
        # Handle labels
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = torch.LongTensor(labels)
        
        # Move to device
        device = next(self.parameters()).device
        features = features.to(device)
        z_scores = z_scores.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(features, z_scores, labels)
        
        # Convert outputs to numpy
        results = {}
        
        # Error detection
        error_probs = torch.softmax(outputs['error_logits'], dim=-1)[:, :, 1]
        results['error_probs'] = error_probs.cpu().numpy()
        results['error_mask'] = (error_probs > threshold).cpu().numpy().astype(int)
        
        # Data repair
        results['repaired'] = outputs['repair_pred'].cpu().numpy()
        
        # Imputation
        results['imputed'] = outputs['impute_pred'].cpu().numpy()
        
        # Label classification
        label_probs = torch.softmax(outputs['label_logits'], dim=-1)
        results['label_probs'] = label_probs.cpu().numpy()
        results['label_pred'] = outputs['label_logits'].argmax(dim=-1).cpu().numpy()
        
        # Noise detection
        noise_probs = torch.softmax(outputs['noise_logits'], dim=-1)[:, 1]
        results['noise_probs'] = noise_probs.cpu().numpy()
        results['noise_mask'] = (noise_probs > threshold).cpu().numpy().astype(int)
        
        # Data valuation
        results['quality_scores'] = outputs['value_pred'].cpu().numpy()
        
        return results
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'n_features': self.n_features,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'n_classes': self.n_classes,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'UNIDQ':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        if device is not None:
            model = model.to(device)
        return model
