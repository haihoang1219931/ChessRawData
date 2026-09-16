import os
import torch
import torchvision.models as models
import openvino as ov

# =========================================================================
# 1. DEFINE MULTI-OUTPUT RESNET INTERPRETATION ARCHITECTURE
# =========================================================================
class MultiOutputResNet18(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Load backbone architecture outline natively
        base_model = models.resnet18(weights=None)
        
        # Isolate the continuous convolutional layers up to Layer 4
        self.features = torch.nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )
        self.avgpool = base_model.avgpool
        self.fc = torch.nn.Linear(512, num_classes) # Your 7-class linear head

    def forward(self, x):
        feat = self.features(x)       # Target Port 0 -> Shape: [1, 512, 60, 105]
        
        # Complete remaining operations to get classification logits
        x = self.avgpool(feat)
        x = torch.flatten(x, 1)
        logits = self.fc(x)           # Target Port 1 -> Shape: [1, 7]
        
        # Enforce exact return order: Port 0 = Spatial Map, Port 1 = Logits
        return feat, logits

# =========================================================================
# 2. LOAD TRAINING WEIGHTS AND EXPORT TO STATIC ONNX
# =========================================================================
def export_onnx(onnx_path, weights_path=None):
    print("[INFO] Initializing Multi-Output ResNet-18 model framework...")
    model = MultiOutputResNet18(num_classes=7)
    
    # Load your custom chess piece weights if a file path is provided
    if weights_path and os.path.exists(weights_path):
        print(f"[INFO] Loading structural weight maps from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        print("[WARN] No weight path found. Exporting base structural layout layers...")
        
    model.eval()

    # Enforce strict static board geometry shape metrics matching your code:
    # Stride dimensions tracking rules: [Batch=1, Channels=3, Height=1920, Width=3360]
    dummy_input = torch.randn(1, 3, 1920, 3360)
    
    print("[INFO] Exporting graph layer modules to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features', 'output'] # Named indexes mapping to Port 0 and Port 1
    )
    print(f"[SUCCESS] ONNX model graph compiled successfully at: {onnx_path}")

# =========================================================================
# 3. DIRECT OPENVINO IR ACCELERATION COMPILATION (ELIMINATES CLI PATH CRASHES)
# =========================================================================
def convert_to_openvino_ir(onnx_path, output_dir):
    print("[INFO] Spawning OpenVINO Core compilation instance...")
    core = ov.Core()
    
    # Read network model directly using OpenVINO front-end parser
    ov_model = ov.convert_model(onnx_path)
    
    # Construct folder destinations cleanly
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    xml_output = os.path.join(output_dir, "chess_piece_resnet18.xml")
    
    print("[INFO] Serializing model graphs and compressing layers to FP16 optimization registers...")
    ov.save_model(ov_model, xml_output, compress_to_fp16=True)
    
    print("\n==================================================")
    print("[SUCCESS] Static OpenVINO IR files built successfully!")
    print(f" -> XML Layer Definitions Map:  {xml_output}")
    print(f" -> BIN Quantized Variable Map: {xml_output.replace('.xml', '.bin')}")
    print("==================================================\n")

if __name__ == "__main__":
    # Define absolute runtime path context parameters
    ONNX_FILE = "chess_piece_resnet18_20260828_100epoch.onnx"
    OUTPUT_FOLDER = "./"
    
    # Optional: If you want to load your trained PyTorch weights file first, point to it here
    # e.g., PYTORCH_WEIGHTS = "/Data/2026/ChessRawData/your_trained_weights.pth"
    PYTORCH_WEIGHTS = None 

    # Execute end-to-end static optimization pipeline loops
    export_onnx(ONNX_FILE, PYTORCH_WEIGHTS)
    convert_to_openvino_ir(ONNX_FILE, OUTPUT_FOLDER)
