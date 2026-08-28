import argparse
import os
import re
import shutil


def multiply_by_sequence(source_file, output_dir, count=8):
    if not os.path.exists(source_file):
        print(f"Error: The source file '{source_file}' does not exist.")
        return

    filename = os.path.basename(source_file)

    # Regex to capture the text part, the input number, and the extension
    # Example: "my_photo-2.jpg" -> base="my_photo", input_num="2", ext="jpg"
    pattern = re.compile(r"^(.+)-(\d+)\.([a-zA-Z0-9]+)$")
    match = pattern.match(filename)

    if not match:
        print(
            f"Error: File name '{filename}' does not match the 'name-number.ext' pattern."
        )
        return

    base_name = match.group(1)
    input_num = int(match.group(2))
    extension = match.group(3)

    # Math to figure out the start index based on the input number
    # If input is 1: (1 - 1) * 8 + 1 = 1   (Generates 1 to 8)
    # If input is 2: (2 - 1) * 8 + 1 = 9   (Generates 9 to 16)
    start_index = (input_num - 1) * count + 1
    end_index = start_index + count

    # Create destination directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Source File: {filename}")
    print(f"Detected Sequence Batch: {input_num}")
    print(f"Generating Indices: {start_index:04d} to {(end_index - 1):04d}")
    print("-" * 50)

    for i in range(start_index, end_index):
        # Format current loop index to 4 digits with zero-padding
        padded_number = str(i).zfill(4)
        new_filename = f"{base_name}-{padded_number}.{extension}"
        destination_path = os.path.join(output_dir, new_filename)

        try:
            shutil.copy2(source_file, destination_path)
            print(f"Created: {new_filename}")
        except Exception as e:
            print(f"  [ERROR] Failed to create copy {new_filename}: {e}")

    print("-" * 50)
    print("Process complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiply a file into 8 copies with mathematically calculated sequence padding."
    )
    parser.add_argument(
        "source", help="Path to the sequential source file (e.g., my_photo-2.jpg)"
    )
    parser.add_argument(
        "folder", help="Path to the destination folder for the copies."
    )

    args = parser.parse_args()
    multiply_by_sequence(args.source, args.folder, count=8)
