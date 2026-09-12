import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# 1. HARDWARE SELECTION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA AUGMENTATION (Optimized for Grayscale 64x64)
# ==========================================
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]
IMAGE_SIZE = (64, 64) # Tiny resolution optimized for speed and embedded deployment

data_transforms = {
    'train': transforms.Compose([
        # Jitter brightness/contrast FIRST while the image is still in RGB format
        transforms.ColorJitter(brightness=0.3, contrast=0.3), 
        # Convert to 1-channel Grayscale
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), # Useful for top-down piece orientation invariance
        transforms.RandomRotation(180),  # Top-down views can be approached from any angle
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Mild translation for 64x64
        transforms.ToTensor(),
        transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD),
        # Lightened erasure scale so it doesn't obliterate fine features at 64x64
        transforms.RandomErasing(p=0.10, scale=(0.01, 0.06), ratio=(0.5, 2.0))
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD)
    ]),
}

# ==========================================
# 3. DATA LOADERS SETUP
# ==========================================
DATA_DIR = './chess_classifier/dataset'

if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory '{DATA_DIR}' not found.")
    sys.exit(1)

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
}

class_names = image_datasets['train'].classes
print(f"Detected Type Classes ({len(class_names)}): {class_names}")

# ==========================================
# 4. MODEL DEFINITION & RESUME FROM CHECKPOINT
# ==========================================
# Initialize a blank resnet18 structure
model = models.resnet18(weights=None)

# Modify the first conv layer to accept 1 channel (Grayscale) instead of 3 (RGB)
original_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=1, 
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias
)

# Reconstruct custom classification head to match your saved layout structure
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(class_names))
)

# LOAD YOUR EXISTING WEIGHT FILE TO RESUME
checkpoint_path = 'chess_piece_resnet18_20260909_1_channel_64x64_2.pth'
if os.path.exists(checkpoint_path):
    print(f"--> Found checkpoint! Loading weights from '{checkpoint_path}'...")
    # map_location ensures weights land on the current device (CPU or GPU) smoothly
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print(f"--> Warning: '{checkpoint_path}' not found. Starting training from scratch.")

# FREEZE EARLY LAYERS & KEEP DEEP LAYERS OPEN
for param in model.parameters():
    param.requires_grad = False

# Ensure our grayscale input conv1 and deep layer4 can adapt to the new 64x64 details
model.conv1.weight.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

# ==========================================
# 5. TARGETED LOSS WEIGHTS & SPEED OPTIMIZER
# ==========================================
class_weights = torch.ones(len(class_names), dtype=torch.float)
for idx, name in enumerate(class_names):
    normalized_name = name.lower()
    if 'bishop' in normalized_name:
        class_weights[idx] = 3.0  # Extra high penalty since color isn't a factor
    elif 'pawn' in normalized_name:
        class_weights[idx] = 1.8  # Heightened penalty to avoid lazy classification shortcuts

class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Track only trainable blocks
trainable_params = [p for p in model.parameters() if p.requires_grad]

# Lowered Learning Rate (lr=1e-5) to fine-tune the resumed weights gently
optimizer = optim.Adam(trainable_params, lr=0.00001, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

# ==========================================
# 6. EARLY STOPPING CONFIGURATION
# ==========================================
epochs = 100
patience = 15  # Increased patience to give the 64x64 grayscale room to stabilize             
patience_counter = 0
best_val_loss = float('inf')

# ==========================================
# 7. TRAINING & VALIDATION LOOP
# ==========================================
print("\n--- Starting Training Workflow ---")
for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == 'val':
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"--> Found better weights! Saved to '{checkpoint_path}'")
            else:
                patience_counter += 1
                print(f"--> No improvement for {patience_counter} consecutive epoch(s).")
                
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered. Target plateau reached at Epoch {epoch + 1}.")
        break

print("\nType-only training workflow finalized successfully.")

# ==========================================
# 8. EXPORT TO ONNX FORMAT (Automated)
# ==========================================
print("\n--- Exporting Finalized Model to ONNX ---")

# Recreate a clean inference architecture configuration instance
onnx_model = models.resnet18(weights=None)
onnx_model.conv1 = nn.Conv2d(
    in_channels=1, 
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias
)
onnx_model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(class_names))
)

# Load the absolute best parameters saved during the run
onnx_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
onnx_model = onnx_model.to(device)
onnx_model.eval() # Set to evaluation mode for deterministic graph output

# Dummy Input configured perfectly for: (Batch Size = 1, Channels = 1, Height = 64, Width = 64)
dummy_input = torch.randn(1, 1, 64, 64, device=device)
onnx_filename = 'chess_piece_resnet18_20260909_1_channel_64x64_2.onnx'

torch.onnx.export(
    onnx_model,
    dummy_input,
    onnx_filename,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={ # Allows deployment runtimes to pass dynamic batch sizes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Successfully generated deployable asset: '{onnx_filename}'\n")
