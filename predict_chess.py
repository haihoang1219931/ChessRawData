import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. SETUP LOGISTICS & HARDWARE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the exact 12 classes in strict alphabetical order (PyTorch default)
class_names = [
    "gen-Bishop", "gen-Empty", "gen-King", "gen-Knight", "gen-Pawn", "gen-Queen", "gen-Rook"
]

# ==========================================
# 2. REBUILD MODEL & LOAD WEIGHTS
# ==========================================
# Initialize the baseline architecture
model = models.resnet18(weights=None) # No default weights needed since we load our own
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Load the saved state dictionary weights onto your hardware
model.load_state_dict(torch.load('chess_piece_resnet18.pth', map_location=device))
model = model.to(device)
model.eval() # Set model to evaluation state (disables dropout/batchnorm)

# ==========================================
# 3. DEFINE INFERENCE TRANSFORMS
# ==========================================
# Use the exact same image resizing and normalization used in your 'val' set
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (240, 240) # ResNet18 expects 240x240 input images
inference_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def predict_image(image_path):
    # Open image using PIL and ensure it is in RGB format
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations and add a fake batch dimension at index 0 (Shape:)
    input_tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    # Disable gradient tracking to save processing memory
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Calculate raw probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Extract highest confidence index and its value
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted_idx.item()]
    confidence_percentage = confidence.item() * 100
    
    return predicted_class, confidence_percentage

# ==========================================
# 5. EXECUTION EXAMPLES
# ==========================================
if __name__ == "__main__":
    # 1. Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run the chess classifier on a target image."
    )
    
    # 2. Define the image path parameter (with your original path as the default)
    parser.add_argument(
        '--image', 
        type=str, 
        default="chess_classifier/dataset/val/black_knight/0000.jpg",
        help="Path to the testing image file"
    )
    
    # 3. Parse the arguments
    args = parser.parse_args()
    TARGET_IMAGE = args.image

    # 4. Safety check: Verify the file actually exists before running your logic
    if not os.path.exists(TARGET_IMAGE):
        print(f"Error: The image file '{TARGET_IMAGE}' does not exist.")
        sys.exit(1)

    print(f"Running classifier on: {TARGET_IMAGE}")
    
    piece_type, confidence = predict_image(TARGET_IMAGE)
    print(f"\nPrediction Result: {piece_type.upper()}")
    print(f"Confidence Level: {confidence:.2f}%")
