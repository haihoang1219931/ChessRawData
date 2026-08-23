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
# 2. DATA AUGMENTATION & PIPELINE
# ==========================================
# Standard normalization parameters for pre-trained ImageNet models
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (240, 240) # ResNet18 expects 240x240 input images
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), # Helps model generalize chess piece angles
        transforms.RandomRotation(15),      # Handles slight board tilts
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ]),
}

# ==========================================
# 3. DATA LOADERS SETUP
# ==========================================
DATA_DIR = './chess_classifier/dataset' # Path matching your directory setup

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=2)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
print(f"Detected Classes ({len(class_names)}): {class_names}")

# ==========================================
# 4. MODEL DEFINTION (TRANSFER LEARNING)
# ==========================================
# Load pre-trained weights for state-of-the-art feature extraction
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final Fully Connected (fc) layer to output 6 classes instead of 1000
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names)) 

model = model.to(device)

# ==========================================
# 5. LOSS & OPTIMIZER CONFIG
# ==========================================
criterion = nn.CrossEntropyLoss() # Standard choice for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 6. TRAINING & VALIDATION LOOP
# ==========================================
epochs = 100

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over batches of images and target labels
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Track gradients only during training phase
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# ==========================================
# 7. SAVE FINAL WEIGHTS
# ==========================================
torch.save(model.state_dict(), 'chess_piece_resnet18.pth')
print("\nModel saved successfully as 'chess_piece_resnet18.pth'")
