import torch
import torch.nn as nn
from torchvision import models

# Define the exact same static architecture used during training
class StaticMiniResNet(nn.Module):
    def __init__(self, num_classes):
        super(StaticMiniResNet, self).__init__()
        # Load structural backbone shell (No weights needed as we load our own .pth file)
        base = models.resnet18(weights=None)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  # 64 channels
        self.layer2 = base.layer2  # 128 channels
        
        # ⚡ HARD HARDWARE ACCELERATION LOCK:
        # Static pooling forces the graph to compile into direct AVX2 vector primitives,
        # completely eliminating OpenVINO JIT loop re-compilations inside your C++ loop.
        self.static_pool = nn.MaxPool2d(kernel_size=30)
        self.flatten = nn.Flatten()
        
        # Enhanced non-linear decision head matching your 7 classes configuration
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

def convert_to_static_onnx():
    # 1. Configuration parameters
    NUM_CLASSES = 7  # Matches your target dataset classes count: b, ., k, n, p, q, r
    IMAGE_SIZE = 240 # Target static resolution model size configuration
    weights_path = 'mini_resnet_chess_240x240_vino_20260918.pth'
    onnx_filename = 'mini_resnet_chess_240x240_vino_20260918.onnx'
    
    print(f"[ONNX EXPORT] Initializing graph structure for {NUM_CLASSES} classes...")
    model = StaticMiniResNet(num_classes=NUM_CLASSES)

    # 2. Load the saved weights dictionary matrix map safely
    print(f"[ONNX EXPORT] Loading checkpoint dictionary from: '{weights_path}'")
    try:
        # map_location='cpu' ensures it opens safely on any machine layout configuration
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"\n[ERROR] Failed to map weights tensor arrays: {e}")
        print("Please ensure your trained .pth file structure matches the static 7-class configuration model head.")
        return

    # 3. Enforce strict evaluation configurations profile
    model.eval() # Hard toggle locks down internal BatchNorm scale vectors and Dropout layers

    # 4. Generate a hardcoded, static structural input matrix tensor
    # Format layout parameters: (Batch_Size, Channels, Height, Width) -> (1, 3, 240, 240)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print(f"[ONNX EXPORT] Compiling fully static trace passes at resolution [{IMAGE_SIZE}x{IMAGE_SIZE}]...")
    
    # 5. Execute structural export pass (Dynamic axes are completely omitted!)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename, 
        export_params=True, 
        opset_version=11,               # Optimal compatibility version layer for OpenCV 4.x/OpenVINO backends
        do_constant_folding=True, 
        input_names=['input'],          # Explicitly maps tracking strings inside C++ setInput passes
        output_names=['output']         # Explicitly maps tracking strings inside C++ forward passes
    )

    print(f"\n[SUCCESS] Production deployment file compiled cleanly and saved as: '{onnx_filename}'")

if __name__ == "__main__":
    convert_to_static_onnx()
