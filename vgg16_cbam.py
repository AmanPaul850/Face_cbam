import torch
import torch.nn as nn
from cbam import CBAM

class LogisticMapping(nn.Module):
    def __init__(self):
        super(LogisticMapping, self).__init__()
        self.a = nn.Parameter(torch.ones(1) * 5.0)  # Upper bound
        self.b = nn.Parameter(torch.zeros(1))       # Lower bound
        self.c = nn.Parameter(torch.ones(1) * 3.0)  # Mid point
        self.d = nn.Parameter(torch.ones(1))        # Slope
        
    def forward(self, x):
        x = torch.clamp(x, min=-50.0, max=50.0)
        return (self.a - self.b) / (1 + torch.exp(-(x - self.c) / (self.d + 1e-7))) + self.b

class VGG16_CBAM_IQA(nn.Module):
    def __init__(self, regression_type='simple'):          
        super(VGG16_CBAM_IQA, self).__init__()
        
        # VGG16 features with CBAM
        self.features = nn.ModuleList([
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # CBAM modules
        self.cbam_modules = nn.ModuleList([
            CBAM(64),    # After block 1
            CBAM(128),   # After block 2
            CBAM(256),   # After block 3
            CBAM(512),   # After block 4
            CBAM(512)    # After block 5
        ])
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size will be (512, 1, 1)
        
        # Fully connected layers to extract features: 512 -> 128 -> 64 -> 32
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )
        
        # Final regression layer mapping the 32-dimensional feature to a single output
        self.regression_layer = nn.Linear(32, 1)
        
        # Add logistic mapping layer
        self.logistic_mapping = LogisticMapping()
        
        # Regression type and parameters
        self.regression_type = regression_type
        if regression_type == 'ridge':
            self.ridge_lambda = 0.01
        elif regression_type == 'elastic':
            self.l1_lambda = 0.01
            self.l2_lambda = 0.01
            
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        cbam_idx = 0
        
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                x = self.cbam_modules[cbam_idx](x)
                cbam_idx += 1
        
        x = self.global_avg_pool(x)  # Apply Global Average Pooling
        x = torch.flatten(x, 1)      # Flatten to 1D vector (size: 512)
        x = self.fc_layers(x)         # Pass through fc layers to get 32-D features
        x = self.regression_layer(x)  # Final regression layer
        
        # Apply logistic mapping
        x = self.logistic_mapping(x)
        
        return x
    
    def get_regularization(self):
        if self.regression_type == 'simple':
            return torch.tensor(0., device=self.regression_layer.weight.device)
        
        elif self.regression_type == 'ridge':
            reg = torch.tensor(0., device=self.regression_layer.weight.device)
            for param in self.regression_layer.parameters():
                reg += self.ridge_lambda * torch.norm(param, p=2) ** 2
            return reg
            
        elif self.regression_type == 'elastic':
            reg = torch.tensor(0., device=self.regression_layer.weight.device)
            for param in self.regression_layer.parameters():
                reg += self.l1_lambda * torch.norm(param, p=1)
                reg += self.l2_lambda * torch.norm(param, p=2) ** 2
            return reg 
