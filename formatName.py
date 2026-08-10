import argparse
import os
import re

def get_sort_key(filename):
    # Extracts the original number from the file name to sort them numerically
    match = re.match(re.compile(r"^(.+)-(\d+)\.([a-zA-Z0-9]+)$"), filename)
    if match:
        return (match.group(1), int(match.group(2)))
    return (filename, 0)

def pad_and_reindex_files(directory, dry_run=True):
    pattern = re.compile(r"^(.+)-(\d+)\.([a-zA-Z0-9]+)$")
    
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    print(f"Scanning: {directory}")
    print("Status: DRY RUN (Previewing sequential re-indexing)" if dry_run else "Status: LIVE MODE (Re-indexing files...)")
    print("-" * 60)

    # 1. Gather all files in the directory
    all_files = os.listdir(directory)
    
    # 2. Sort them numerically so the original order is preserved before re-indexing
    all_files.sort(key=get_sort_key)

    # 3. Create an internal counter for sequential re-indexing
    sequence_counter = 1
    renamed_count = 0

    # 4. Process the sorted files
    for filename in all_files:
        match = pattern.match(filename)
        if match:
            base_name = match.group(1)
            extension = match.group(3)
            
            # Format the counter instead of using the old number string
            padded_number = str(sequence_counter).zfill(4)
            new_filename = f"{base_name}-{padded_number}.{extension}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # Print preview message
            if filename != new_filename:
                print(f"Re-index: {filename} -> {new_filename}")
                renamed_count += 1
            else:
                print(f"Unchanged: {filename} (Already fits sequence)")
            
            if not dry_run:
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f"  [ERROR] Could not rename {filename}: {e}")
                    # Do not advance counter if rename fails on live run
                    continue
            
            # Advance the counter for the next matched file
            sequence_counter += 1

    print("-" * 60)
    print(f"Process complete. Files updated or repositioned: {renamed_count}")
    if dry_run and renamed_count > 0:
        print("To apply these sequential changes, run the command again with the --run flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index files into a strict 4-digit sequence with zero padding.")
    parser.add_argument("folder", help="Path to the target folder containing files.")
    parser.add_argument("--run", action="store_false", dest="dry_run", 
                        help="Execute the renaming process. Without this flag, it only previews changes.")

    args = parser.parse_args()
    pad_and_reindex_files(args.folder, dry_run=args.dry_run)
