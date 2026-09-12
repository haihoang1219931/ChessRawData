import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
# ==========================================
# 8. EXPORT TO ONNX FORMAT
# ==========================================
print("\n--- Exporting Model to ONNX ---")

# 1. Initialize a model instance with the same grayscale architecture
onnx_model = models.resnet18()
class_names = [
    "gen-Bishop", "gen-Empty", "gen-King", "gen-Knight", "gen-Pawn", "gen-Queen", "gen-Rook"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Recreate the exact 1-channel grayscale modification used during training
original_conv = onnx_model.conv1
onnx_model.conv1 = nn.Conv2d(
    in_channels=1, # Match grayscale input channel
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias
)

# Recreate the custom final classification layer
num_features = onnx_model.fc.in_features
onnx_model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(class_names))
)

# 2. Load the best weights saved during training
weights_path = 'chess_piece_resnet18_20260909_1_channel_64x64_2.pth'
onnx_model.load_state_dict(torch.load(weights_path, map_location=device))
onnx_model = onnx_model.to(device)
onnx_model.eval() # Always set to evaluation mode before exporting

# 3. Create a dummy input tensor matching your new parameters: 
# (Batch Size = 1, Channels = 1, Height = 64, Width = 64)
dummy_input = torch.randn(1, 1, 64, 64, device=device)

# 4. Execute the ONNX export
onnx_filename = 'chess_piece_resnet18_20260909_1_channel_64x64_2.onnx'
torch.onnx.export(
    onnx_model,                  # The trained PyTorch model
    dummy_input,                 # Sample input tensor with correct dimensions
    onnx_filename,               # Output file path
    export_params=True,          # Store the trained parameter weights inside the file
    opset_version=12,            # Standard, widely compatible ONNX version
    do_constant_folding=True,    # Optimizes the network by folding constant nodes
    input_names=['input'],       # Name of the input layer node
    output_names=['output'],     # Name of the output layer node
    dynamic_axes={               # Allow flexible batch sizing during deployment inference
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model successfully exported to deployable format: '{onnx_filename}'")