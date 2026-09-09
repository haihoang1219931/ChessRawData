import os
import sys
import argparse
import numpy as np
import onnxruntime as ort  # Added for ONNX inference
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. SETUP LOGISTICS & ENVIRONMENT
# ==========================================
# Define the exact classes in strict alphabetical order (PyTorch default)
class_names = [
    "gen-Bishop", "gen-Empty", "gen-King", "gen-Knight", "gen-Pawn", "gen-Queen", "gen-Rook"
]

# ==========================================
# 2. LOAD ONNX RUNTIME SESSION
# ==========================================
onnx_model_path = 'chess_piece_resnet18_pawn_bishop.onnx'

if not os.path.exists(onnx_model_path):
    print(f"Error: ONNX model file '{onnx_model_path}' not found. Please export it first.")
    sys.exit(1)

# Initialize the ONNX inference session (automatically picks CUDA if available, else CPU)
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Get the exact name of the input node expected by the ONNX model (e.g., 'input')
input_name = ort_session.get_inputs()[0].name

# ==========================================
# 3. DEFINE INFERENCE TRANSFORMS (Grayscale + 120x120)
# ==========================================
# Replaced ImageNet RGB stats with standard single-channel Grayscale stats
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]
IMAGE_SIZE = (120, 120) # Updated from 240x240 to 120x120 for the ONNX model graph

inference_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD)
])

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def predict_image(image_path):
    # Open image using PIL and force convert your RGB image to 1-channel Grayscale ('L')
    image = Image.open(image_path).convert('L')
    
    # Apply transformations and add a fake batch dimension at index 0 (Shape becomes:)
    input_tensor = inference_transforms(image).unsqueeze(0)
    
    # Convert PyTorch Tensor to a standard NumPy array matching the ONNX signature
    input_numpy = input_tensor.numpy()
    
    # Run inference through ONNX Runtime
    outputs = ort_session.run(None, {input_name: input_numpy})
    
    # Extract the raw logits tensor array [batch_size, num_classes]
    logits = outputs[0][0]
    
    # Calculate raw probabilities using Softmax in NumPy
    exp_logits = np.exp(logits - np.max(logits)) # Subtraction avoids numerical overflow
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Extract highest confidence index and its value
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
        
    predicted_class = class_names[predicted_idx]
    confidence_percentage = confidence * 100
    
    return predicted_class, confidence_percentage

# ==========================================
# 5. EXECUTION EXAMPLES
# ==========================================
if __name__ == "__main__":
    # 1. Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run the chess classifier ONNX model on a target image."
    )
    
    # 2. Define the image path parameter
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

    print(f"Running ONNX classifier on: {TARGET_IMAGE}")
    
    piece_type, confidence = predict_image(TARGET_IMAGE)
    print(f"\nPrediction Result: {piece_type.upper()}")
    print(f"Confidence Level: {confidence:.2f}%")