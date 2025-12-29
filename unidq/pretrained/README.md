# Pre-trained Weights

This directory is reserved for storing pre-trained UNIDQ model weights.

## Usage

To load a pre-trained model:

```python
from unidq import UNIDQ

model = UNIDQ.from_pretrained("path/to/pretrained/weights")
```

## Directory Structure

```
pretrained/
├── README.md              # This file
├── model_v1/             # Example model version 1
│   ├── config.json       # Model configuration
│   └── pytorch_model.bin # Model weights
└── model_v2/             # Example model version 2
    ├── config.json
    └── pytorch_model.bin
```

## Sharing Weights

When sharing pre-trained weights, ensure to include:
1. `config.json` - Model configuration
2. `pytorch_model.bin` - PyTorch state dictionary
3. Documentation about training data and performance

## Downloads

Pre-trained weights will be made available at:
- GitHub Releases: [Coming Soon]
- Hugging Face Hub: [Coming Soon]
