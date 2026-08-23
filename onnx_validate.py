import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

# ==========================================
# 1. SETUP LOGISTICS & CONFIGURATION
# ==========================================
ONNX_MODEL_PATH = "chess_piece_resnet18.onnx"
TARGET_IMAGE = "chess_classifier/dataset/train/gen-Empty/1.jpg"

CLASS_NAMES = [
    "gen-Bishop", "gen-Empty", "gen-King", "gen-Knight", "gen-Pawn", "gen-Queen", "gen-Rook"
]

# ImageNet normalization stats
IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = (240, 240)

# ==========================================
# 2. VALIDATION CHECKS
# ==========================================
if not os.path.exists(ONNX_MODEL_PATH):
    print(f"Error: Model file '{ONNX_MODEL_PATH}' not found.")
    sys.exit(1)

if not os.path.exists(TARGET_IMAGE):
    print(f"Error: Target image '{TARGET_IMAGE}' does not exist.")
    sys.exit(1)

# ==========================================
# 3. LOAD THE ONNX INTERPRETER
# ==========================================
print("Loading ONNX model using ONNX Runtime...")
# Automatically falls back to CPU if your system's CUDA driver is too old
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

# Extract expected input layer name from the compiled graph configuration
input_name = session.get_inputs()[0].name

# ==========================================
# 4. PREPROCESS IMAGE (PURE NUMPY / PIL)
# ==========================================
# 1. Load image and ensure it has 3 channels (RGB)
img = Image.open(TARGET_IMAGE).convert('RGB')

# 2. Resize to your model dimension (240x240)

# 3. Convert to Numpy Array and scale pixels to [0.0, 1.0]
img_data = np.array(img, dtype=np.float32) / 255.0

# 4. Apply Channel-wise Normalization: (Pixel - Mean) / Std
img_data = (img_data - IMAGE_NET_MEAN) / IMAGE_NET_STD

# 5. Permute from [Height, Width, Channels] to [Channels, Height, Width]
img_data = img_data.transpose(2, 0, 1)

# 6. Add fake batch dimension at index 0 -> [1, Channels, Height, Width]
input_tensor = np.expand_dims(img_data, axis=0)

# ==========================================
# 5. RUN EXECUTION ENGINE
# ==========================================
# Run inference by passing a dictionary mapping input layer name to the matrix array
outputs = session.run(None, {input_name: input_tensor})
raw_predictions = outputs[0]  # Shape: [1, num_classes]

# Calculate Softmax probabilities
exp_outputs = np.exp(raw_predictions - np.max(raw_predictions))
probabilities = exp_outputs / np.sum(exp_outputs)

# Extract highest match
predicted_idx = np.argmax(probabilities)
confidence = probabilities[0][predicted_idx] * 100

# ==========================================
# 6. OUTPUT RESULTS
# ==========================================
print(f"\nRunning ONNX classifier on: {TARGET_IMAGE}")
print(f"Prediction Result: {CLASS_NAMES[predicted_idx].upper()}")
print(f"Confidence Level: {confidence:.2f}%")
