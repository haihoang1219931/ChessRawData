import torch
import torch.nn as nn
from torchvision import models

# 1. Exact replication of your static training architecture
class StaticMiniResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(StaticMiniResNet, self).__init__()
        base = models.resnet18(weights=None)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  
        self.layer2 = base.layer2  
        
        self.static_pool = nn.MaxPool2d(kernel_size=30)
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.static_pool(x) 
        x = self.flatten(x)
        x = self.fc(x)
        return x

def export_perfect_onnx():
    NUM_CLASSES = 7
    IMAGE_SIZE = 240
    weights_path = 'mini_resnet_chess_240x240_20260917.pth'
    onnx_filename = 'mini_resnet_chess_240x240_20260917.onnx'
    
    model = StaticMiniResNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    # ⚡ HARD HARDWARE FIX: Trace the model with a base batch size of 1
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("[ONNX] Compiling structured vector layout profiles...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        # ⚡ CRITICAL ACCELERATION AXES: We must explicitly name the batch axis 
        # as dynamic so OpenVINO's compiler can linearize vector grids cleanly!
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"[ONNX SUCCESS] Export complete -> {onnx_filename}")

if __name__ == "__main__":
    export_perfect_onnx()
