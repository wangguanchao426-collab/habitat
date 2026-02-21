# CNN Extension for Habitat Operator Runtime Prediction
A CNN-based extension module for operator runtime prediction, aligned with the original MLP interface in the Habitat project.

## Overview
This project extends the Habitat project with CNN-based models (LinearCNN/LSTMCNN/Conv2DCNN/BMMCNN) for operator runtime prediction. The core logic reuses MLPBase from the original project to ensure interface consistency, while adding CNN backbone support for potential feature extraction optimization.

### Key Features
- 4 CNN variants (LinearCNN/LSTMCNN/Conv2DCNN/BMMCNN) aligned with MLP naming convention
- Dynamically calculate CNN output dimension via random tensor (PyTorch)
- Consistent API with original MLP modules for easy integration

### Important Notes
1. Experimental Status: This CNN module is a **reserved extension** and has **NOT been fully tested/validated** with real datasets or runtime prediction tasks. The code is provided as a structural reference only.
2. AI Assistance: Core code logic was assisted by AI tools, with manual adjustment for interface alignment and readability.

## Installation
### Prerequisites
- Python 3.7+
- PyTorch ≥ 1.8.0
- NumPy ≥ 1.21.0

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Example
```python
from habitat.analysis.mlp import LinearCNN

# Initialize CNN model (example parameters)
model = LinearCNN(layers=3, layer_size=128, total_dim=256)

# Note: No actual training/inference logic is tested
# This is just a structural example
```

## Acknowledgments
This project is forked from [Original Habitat Repository URL] (replace with the original repo link)
Reuses MLPBase module from the original Habitat project for interface consistency
AI tools were used to assist in code development

## License
MIT License (see LICENSE file for details)
