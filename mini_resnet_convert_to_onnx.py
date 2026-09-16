import torch
import torch.nn as nn
from torchvision import models

def export_pth_to_onnx():
    # 1. Configuration parameters
    NUM_CLASSES = 7  # Matches your target dataset classes count
    IMAGE_SIZE = 64 # Target model size configuration matching OpenCV
    weights_path = 'mini_resnet_chess_64x64.pth'
    onnx_filename = 'mini_resnet_chess_64x64.onnx'
    
    print(f"Initializing structural graph framework for {NUM_CLASSES} classes...")

    # 2. Re-create the EXACT Mini-ResNet architecture used during training
    full_resnet = models.resnet18(weights=None) # No default ImageNet load needed since we overwrite it

    model = nn.Sequential(
        full_resnet.conv1,
        full_resnet.bn1,
        full_resnet.relu,
        full_resnet.maxpool,
        full_resnet.layer1,     
        full_resnet.layer2,     
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    # Re-attach the exact non-linear classification head mapping
    num_features = 128 
    model.add_module("fc", nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, NUM_CLASSES)
    ))

    # 3. Load the saved weights dictionary matrix map
    print(f"Loading weights array maps from: '{weights_path}'")
    try:
        # map_location='cpu' ensures it opens safely even if trained on a system with broken CUDA drivers
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"\n[ERROR] Failed to map weights tensor arrays: {e}")
        print("Please ensure your trained .pth file structure matches the 7-class configuration model head.")
        return

    # 4. Enforce strict evaluation configurations profile
    model.eval() # Hard toggle locks down internal BatchNorm scale vectors and Dropout layers

    # 5. Generate dummy structural verification input matrix tensor tracking trace paths
    # Format layout parameters: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print(f"Compiling tracking graphs trace passes at resolution [{IMAGE_SIZE}x{IMAGE_SIZE}]...")
    
    # 6. Execute structural export pass
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename, 
        export_params=True, 
        opset_version=11,               # Optimal compatibility version layer for OpenCV 4.x backends
        do_constant_folding=True, 
        input_names=['input'],          # Explicitly maps tracking strings inside C++ setInput passes
        output_names=['output'],         # Explicitly maps tracking strings inside C++ forward passes
        dynamic_axes={
            'input': {0: 'batch_size'}, # Unlocks multi-square parallel batch execution sizes in OpenCV
            'output': {0: 'batch_size'}
        }
    )

    print(f"\n[SUCCESS] Production deployment file compiled cleanly and saved as: '{onnx_filename}'")

if __name__ == "__main__":
    export_pth_to_onnx()
