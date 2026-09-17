import os
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
# 2. DATA AUGMENTATION & PIPELINE (Optimized to fight Overfitting)
# ==========================================
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (32, 32) # Optimized lightweight footprint grid size

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),             
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]),
}

# ==========================================
# 3. DATA LOADERS SETUP (Optimized for Speed)
# ==========================================
DATA_DIR = './chess_classifier/dataset' 

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
}

class_names = image_datasets['train'].classes
print(f"Detected Classes ({len(class_names)}): {class_names}")

# ==========================================
# 4. MODEL DEFINITION & CHECKPOINT RESUME
# ==========================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(class_names))
)

model = model.to(device)

# ⚡ UPDATED: Checkpoint checkpoint loading module
# Looks for existing weights from previous training sessions to resume progress
WEIGHTS_PATH = 'chess_piece_resnet18_20260917_32x32_3channels.pth'

if os.path.exists(WEIGHTS_PATH):
    print(f"\n[CHECKPOINT] Found existing weights file at '{WEIGHTS_PATH}'")
    print("--> Loading saved parameters matrix to resume progressive training...")
    try:
        # map_location ensures safety if switching between CUDA and CPU topologies
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("[CHECKPOINT SUCCESS] State dictionary matched cleanly.\n")
    except Exception as e:
        print(f"[CHECKPOINT WARNING] Error parsing weights file: {e}")
        print("--> Proceeding with baseline initialization parameters instead.\n")
else:
    print(f"\n[INITIALIZATION] No previous weights file found at '{WEIGHTS_PATH}'")
    print("--> Starting fresh training pass using pre-trained ImageNet parameters.\n")

# ==========================================
# 5. LOSS, OPTIMIZER & AMP CONFIG (Added Weight Decay & AMP)
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

# ==========================================
# 6. EARLY STOPPING CONFIGURATION
# ==========================================
epochs = 1000
patience = 100               
patience_counter = 0
best_val_loss = float('inf')

# ==========================================
# 7. TRAINING & VALIDATION LOOP
# ==========================================
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
                torch.save(model.state_dict(), WEIGHTS_PATH)
                print(f"--> Found better weights! Saved to '{WEIGHTS_PATH}'")
            else:
                patience_counter += 1
                print(f"--> No improvement for {patience_counter} consecutive epoch(s).")
                
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered. Target plateau reached at Epoch {epoch + 1}.")
        break

print("\nTraining workflow finalized successfully.")

# ==========================================
# 8. AUTOMATED STATIC ONNX EXPORT MODULE
# ==========================================
print("\n=== Initializing Static ONNX Export ===")
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

model.eval()

# Generate hardcoded static single-image dimension reference tracking metrics
# Format layout parameters: (Batch_Size, Channels, Height, Width) -> (1, 3, 32, 32)
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
onnx_filename = "chess_piece_resnet18_20260917_32x32_3channels.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_filename, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output']
    # Dynamic axes are completely omitted to ensure maximum AVX2 compilation speed
)
print(f"[ONNX SUCCESS] Production deployment file compiled cleanly as: '{onnx_filename}'")
