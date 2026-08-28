import os
import torch
import torch.nn as nn
from torchvision import datasets, models

# ==========================================
# 1. SETUP HARDWARE & PATHS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './chess_classifier/dataset'
WEIGHTS_PATH = 'chess_piece_resnet18_20260828_100epoch.pth'
OUTPUT_ONNX_PATH = 'chess_piece_resnet18_20260828_100epoch.onnx'

# ==========================================
# 2. DYNAMICALLY DETECT CLASS COUNT
# ==========================================
if not os.path.exists(os.path.join(DATA_DIR, 'train')):
    raise FileNotFoundError(f"Could not find training directory at {DATA_DIR}/train. "
                            f"Ensure your dataset folder is present to match class counts.")

# Use ImageFolder just to grab the exact class configuration used during training
temp_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'))
class_names = temp_dataset.classes
num_classes = len(class_names)
print(f"Detected {num_classes} classes from dataset: {class_names}")

# ==========================================
# 3. REBUILD THE NETWORK ARCHITECTURE
# ==========================================
# Initialize an empty ResNet18 model (no default ImageNet weights needed)
model = models.resnet18(weights=None)

# Remap the final layer to your precise class count
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Identity(), 
    nn.Linear(num_features, num_classes)
)

# Load your custom trained weights
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Could not find weights file '{WEIGHTS_PATH}'. Please run training first.")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model = model.to(device)

# Crucial step: Set model to evaluation mode to disable dropout/batchnorm updates
model.eval()

print("Model architecture matched and weights loaded successfully.")

# ==========================================
# 4. EXPORT TO ONNX
# ==========================================
# Create a dummy image matrix matching your training input shape: 
# [Batch size: 1, Channels: 3, Height: 240, Width: 240]
dummy_input = torch.randn(1, 3, 240, 240).to(device)

print(f"Exporting model to ONNX format at '{OUTPUT_ONNX_PATH}'...")

with torch.no_grad():
    torch.onnx.export(
        model, 
        dummy_input, 
        OUTPUT_ONNX_PATH, 
        export_params=True,        # Store the trained weights inside the ONNX file
        opset_version=12,          # High compatibility standard for OpenCV DNN
        do_constant_folding=True,  # Optimizes graph structural processing speed
        input_names=['input'],     # Label for the entry data tensor
        output_names=['output'],   # Label for the prediction vector tensor
        dynamic_axes={             # Allows flexible batch sizes during live C++ runtime execution
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )

print("ONNX conversion completed perfectly!")
