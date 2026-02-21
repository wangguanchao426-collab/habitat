"""
This module is reserved for future extension (comparison with MLP).
The model structure is designed to be compatible with LinearMLP's input/output interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLPBase  # Reuse MLPBase as sub-module 

__all__ = ["LinearCNN", "LSTMCNN", "Conv2DCNN", "BMMCNN"]  # Align with MLP's model naming


class BaseCNN(nn.Module):
    """
    Base CNN class for operator runtime prediction.
    Shared structure for all operator-specific CNN models.
    """
    def __init__(self, layers, layer_size, feature_dim):
        """
        Initialize BaseCNN.
        Args:
            layers: Number of layers in MLPBase 
            layer_size: Dimension of each MLP layer 
            feature_dim: Number of operator-specific features（Linear=4/LSTM=7/Conv2D=8/BMM=4）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.device_param_dim = 4  #  mem/mem_bw/num_sm/single
        self.total_input_dim = feature_dim + self.device_param_dim

        # CNN backbone: extract spatial features from 1D structured data
        # Reshape 1D input (total_input_dim) to 2D: [1, H, W] (H*W = total_input_dim)
        self.conv_backbone = nn.Sequential(
            # Conv layer 1: extract local features
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer 2: enhance feature representation
            nn.Conv2d(16, 32, kernel_size=1),  # 1x1 conv for dimension adjustment
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Prevent overfitting
        )

        # Calculate output size of CNN backbone (dynamic for different input dims)
        self.h = 2 if self.total_input_dim % 2 ==0 else 3
        self.w = self.total_input_dim // self.h
        cnn_out_dim = self.conv_backbone(torch.randn(1,1,self.h,self.w)).numel()
        
        # Project CNN features to MLP layer size
        self.fc_cnn = nn.Linear(cnn_out_dim, layer_size)
        # Reuse MLPBase (consistent with LinearMLP's structure)
        self.mlp = MLPBase(layers, layer_size)
        # Final prediction layer (output 1D runtime)
        self.fc_out = nn.Linear(layer_size, 1)

    def forward(self, x):
        """
        Forward pass of BaseCNN.
        Args:
            x: Input tensor (batch_size, total_input_dim)
        Returns:
            1D runtime prediction (batch_size, 1)
        """
        # Reshape 1D structured data to 2D for CNN
        x = x.view(x.size(0), 1, self.h, self.w)
        
        # CNN feature extraction
        x = self.conv_backbone(x)
        # Flatten CNN output to 1D
        x = torch.flatten(x, start_dim=1)
        
        # Project to MLP layer size
        x = self.fc_cnn(x)
        x = F.relu(x)
        
        # Reuse MLPBase (consistent with LinearMLP)
        x = self.mlp(x)
        
        # Final prediction
        x = self.fc_out(x)
        return x


class LinearCNN(BaseCNN):
    """CNN model for Linear operator runtime prediction (align with LinearMLP)."""
    def __init__(self, layers, layer_size):
        # Operator-specific features: ["bias", "batch", "in_features", "out_features"]
        feature_dim = 4
        super().__init__(layers, layer_size, feature_dim)
        self.features = ["bias", "batch", "in_features", "out_features"]  # Align with LinearMLP


class LSTMCNN(BaseCNN):
    """CNN model for LSTM operator runtime prediction (align with LSTMMLP)."""
    def __init__(self, layers, layer_size):
        # Operator-specific features for LSTM
        feature_dim = 7
        super().__init__(layers, layer_size, feature_dim)
        self.features = ["bias", "bidirectional", "batch", "seq_len", 
                         "input_size", "hidden_size", "num_layers"]  # Align with LSTMMLP


class Conv2DCNN(BaseCNN):
    """CNN model for Conv2D operator runtime prediction (align with Conv2DMLP)."""
    def __init__(self, layers, layer_size):
        # Operator-specific features for Conv2D
        feature_dim = 8
        super().__init__(layers, layer_size, feature_dim)
        self.features = ["bias", "batch", "image_size", "in_channels", 
                         "out_channels", "kernel_size", "stride", "padding"]  # Align with Conv2DMLP


class BMMCNN(BaseCNN):
    """CNN model for BMM operator runtime prediction (align with BMMMLP)."""
    def __init__(self, layers, layer_size):
        # Operator-specific features for BMM
        feature_dim = 4
        super().__init__(layers, layer_size, feature_dim)
        self.features = ["batch", "left", "middle", "right"]  # Align with BMMMLP