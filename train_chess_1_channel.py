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
# 2. DATA AUGMENTATION (Modified for Grayscale & 120x120)
# ==========================================
# Replaced ImageNet RGB stats with standard single-channel Grayscale stats
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]
IMAGE_SIZE = (120, 120) # Updated from 240x240 to 120x120

data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Converts 3-channel RGB to 1-channel Grayscale
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(180),  
        transforms.ColorJitter(brightness=0.4, contrast=0.4), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Converts 3-channel RGB to 1-channel Grayscale
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD)
    ]),
}

# ==========================================
# 3. DATA LOADERS SETUP
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
print(f"Detected Type Classes ({len(class_names)}): {class_names}")

# ==========================================
# 4. MODEL DEFINITION (Fine-Tuning Deep Layers)
# ==========================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# --- NEW: Modify ResNet18 to accept 1 input channel instead of 3 ---
original_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=1, # Updated for grayscale channel
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias
)
# Re-use pretrained weights by averaging across the 3 original RGB channels
with torch.no_grad():
    model.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

# Freeze all early layers to keep basic geometric edge detectors
for param in model.parameters():
    param.requires_grad = False

# Make sure our new grayscale conv1 layer is trainable
model.conv1.weight.requires_grad = True

# Unfreeze layer4 to let ResNet specialize on the fine cuts of your specific pieces
for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(class_names))
)

model = model.to(device)

# ==========================================
# 5. TARGETED LOSS WEIGHTS & SPEED OPTIMIZER
# ==========================================
class_weights = torch.ones(len(class_names), dtype=torch.float)
for idx, name in enumerate(class_names):
    normalized_name = name.lower()
    if 'bishop' in normalized_name:
        class_weights[idx] = 3.0  
    elif 'pawn' in normalized_name:
        class_weights[idx] = 1.8  

class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Track only trainable blocks
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.00005, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

# ==========================================
# 6. EARLY STOPPING CONFIGURATION
# ==========================================
epochs = 100
patience = 20               
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
                torch.save(model.state_dict(), 'chess_piece_resnet18.pth')
                print("--> Found better weights! Saved to 'chess_piece_resnet18.pth'")
            else:
                patience_counter += 1
                print(f"--> No improvement for {patience_counter} consecutive epoch(s).")
                
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered. Target plateau reached at Epoch {epoch + 1}.")
        break

print("\nType-only training workflow finalized successfully.")
