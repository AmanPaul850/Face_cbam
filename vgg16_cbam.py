import torch
import torch.nn as nn
from torchvision import models
from cbam import CBAM  # Ensure your CBAM module is implemented in cbam.py

class LogisticMapping(nn.Module):
    def __init__(self):
        super(LogisticMapping, self).__init__()
        # Parameters to constrain the output range.
        self.a = nn.Parameter(torch.ones(1) * 5.0)  # Upper bound
        self.b = nn.Parameter(torch.zeros(1))       # Lower bound
        self.c = nn.Parameter(torch.ones(1) * 3.0)    # Midpoint
        self.d = nn.Parameter(torch.ones(1))          # Slope
        
    def forward(self, x):
        x = torch.clamp(x, min=-50.0, max=50.0)
        # Apply a logistic function to squash the output.
        return (self.a - self.b) / (1 + torch.exp(-(x - self.c) / (self.d + 1e-7))) + self.b

class VGG16_CBAM_IQA(nn.Module):
    def __init__(self, regression_type='simple', pretrained=True):
        """
        This model uses a pretrained VGG16 network (with batch normalization)
        augmented with CBAM modules. After global average pooling, the resulting
        512-dimensional feature vector is fed directly into a single linear layer
        (i.e. linear regression) to predict the face quality score.
        """
        super(VGG16_CBAM_IQA, self).__init__()
        self.regression_type = regression_type

        # Load a pretrained VGG16_BN backbone.
        if pretrained:
            vgg = models.vgg16_bn(pretrained=True)
            features = list(vgg.features.children())
            # Optionally freeze early layers to preserve pretrained features.
            for layer in features[:20]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            # (Implement model from scratch if necessary.)
            features = []  # Fill in if not using pretrained weights.
        
        self.features = nn.Sequential(*features)
        
        # Determine indices at which MaxPool occurs. CBAM is applied after these indices.
        self.cbam_indices = [6, 13, 23, 33, 43] if pretrained else [4, 9, 16, 23, 30]
        
        # Create a CBAM module for each block (channels correspond to VGG16 progression).
        cbam_channels = [64, 128, 256, 512, 512]
        self.cbam_modules = nn.ModuleList([CBAM(ch) for ch in cbam_channels])
        
        # Global Average Pooling to reduce the feature map to size (B, 512, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Linear regression head: map the 512-D features (after flattening) to a single output.
        self.regression_layer = nn.Linear(512, 1)
        
        # Logistic mapping to squash and constrain the predicted score.
        self.logistic_mapping = LogisticMapping()
        
        # Set regularization parameters if using ridge or elastic regression.
        if regression_type == 'ridge':
            self.ridge_lambda = 0.01
        elif regression_type == 'elastic':
            self.l1_lambda = 0.01
            self.l2_lambda = 0.01
        
        self._initialize_new_weights()
        
    def _initialize_new_weights(self):
        # Initialize only the new (non-pretrained) layers.
        nn.init.xavier_normal_(self.regression_layer.weight)
        if self.regression_layer.bias is not None:
            nn.init.constant_(self.regression_layer.bias, 0)
    
    def forward(self, x):
        cbam_counter = 0
        out = x
        # Forward through the VGG16 backbone, applying CBAM after each MaxPool.
        for idx, layer in enumerate(self.features):
            out = layer(out)
            if idx in self.cbam_indices:
                out = self.cbam_modules[cbam_counter](out)
                cbam_counter += 1
        
        out = self.global_avg_pool(out)    # Shape becomes (B, 512, 1, 1)
        out = torch.flatten(out, 1)          # Flatten to (B, 512)
        out = self.regression_layer(out)     # Linear regression mapping
        out = self.logistic_mapping(out)     # Constrain the output range
        return out
    
    def get_regularization(self):
        """Compute and return the regularization term."""
        if self.regression_type == 'simple':
            return torch.tensor(0., device=self.regression_layer.weight.device)
        elif self.regression_type == 'ridge':
            reg = torch.tensor(0., device=self.regression_layer.weight.device)
            for param in self.regression_layer.parameters():
                reg += self.ridge_lambda * (torch.norm(param, p=2) ** 2)
            return reg
        elif self.regression_type == 'elastic':
            reg = torch.tensor(0., device=self.regression_layer.weight.device)
            for param in self.regression_layer.parameters():
                reg += self.l1_lambda * torch.norm(param, p=1)
                reg += self.l2_lambda * (torch.norm(param, p=2) ** 2)
            return reg
