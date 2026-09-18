import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

# ==========================================
# 1. HARDWARE SELECTION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA AUGMENTATION (0-Normalization Optimization)
# ==========================================
IMAGE_SIZE = (240, 240)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),             
        transforms.ToTensor(), # Scales pixels to [0.0, 1.0] only. No Normalize() loops!
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
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
# 4. MODEL DEFINITION (Static High-Speed Mini-ResNet)
# ==========================================
class StaticMiniResNet(nn.Module):
    def __init__(self, num_classes):
        super(StaticMiniResNet, self).__init__()
        # Load pre-trained foundation shapes
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  # 64 channels
        self.layer2 = base.layer2  # 128 channels
        
        # ⚡ AVX2 HARDWARE ACCELERATION FIX:
        # At 240x240 input, the output spatial grid leaving layer2 is exactly 30x30.
        # Replacing AdaptiveAvgPool2d with a static MaxPool2d completely eliminates 
        # OpenVINO dynamic loop parsing. This compiles directly into hardcoded CPU instructions.
        self.static_pool = nn.MaxPool2d(kernel_size=30)
        self.flatten = nn.Flatten()
        
        # Enhanced non-linear head matching your 7-class configuration
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
        x = self.static_pool(x) # Static vector layer execution pass
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = StaticMiniResNet(num_classes=len(class_names))

# --- LAYER FREEZING ---
# Freeze early universal edge layers completely to preserve ImageNet shapes
for param in model.conv1.parameters(): param.requires_grad = False
for param in model.bn1.parameters(): param.requires_grad = False
for param in model.layer1.parameters(): param.requires_grad = False

# Keep layer2 and the fc head open to learn custom chess silhouettes
for param in model.layer2.parameters(): param.requires_grad = True
for param in model.fc.parameters(): param.requires_grad = True

model = model.to(device)

# ==========================================
# 5. TARGETED LOSS WEIGHTS & OPTIMIZER
# ==========================================
# Keep custom class weighting strategy to ensure pawns/bishops converge smoothly
class_weights = torch.ones(len(class_names), dtype=torch.float)
for idx, name in enumerate(class_names):
    normalized_name = name.lower()
    if 'bishop' in normalized_name:
        class_weights[idx] = 3.0  
    elif 'pawn' in normalized_name:
        class_weights[idx] = 1.8  

class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Only track trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.001, weight_decay=1e-4)

# Modernized, non-deprecated GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# ==========================================
# 6. EARLY STOPPING CONFIGURATION
# ==========================================
epochs = 1000                 # Reduced because single-layer head adjustments converge fast
patience = 100               
patience_counter = 0
best_val_loss = float('inf')
best_weights_path = 'mini_resnet_chess_240x240_vino_20260918.pth'

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
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
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
                torch.save(model.state_dict(), best_weights_path)
                print(f"--> Found better weights! Saved to '{best_weights_path}'")
            else:
                patience_counter += 1
                print(f"--> No improvement for {patience_counter} consecutive epoch(s).")
                
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered. Target plateau reached.")
        break

print("\nTraining workflow finalized successfully.")

# ==========================================
# 8. AUTOMATED STATIC ONNX EXPORT
# ==========================================
print("\n=== Initializing Static ONNX Export ===")
model.load_state_dict(torch.load(best_weights_path))
model.eval()

# ⚡ HARD SPEED FIX: Dynamic axes completely removed! 
# We target a hard static shape of [1, 3, 240, 240] so OpenVINO can fully serialize memory jumps.
dummy_input = torch.randn(1, 3, 240, 240).to(device)
onnx_filename = "mini_resnet_chess_240x240_vino_20260918.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_filename, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output']
)
print(f"Production file compiled cleanly and saved as: '{onnx_filename}'")
