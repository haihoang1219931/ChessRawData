import os

# Define the root dataset directory
DATASET_DIR = "dataset"

# Define the sub-splits
splits = ["train", "val"]

# Define the 12 distinct chess piece color + type classes
classes = [
    "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
    "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
]

print("Starting directory creation pipeline...")

# Loop through each structural combination
for split in splits:
    for chess_class in classes:
        # Construct the target path (e.g., 'dataset/train/black_pawn')
        target_path = os.path.join(DATASET_DIR, split, chess_class)
        
        # Create folder if it doesn't already exist
        os.makedirs(target_path, exist_ok=True)

print(f"Success! Folder tree created inside local directory: ./{DATASET_DIR}/")
