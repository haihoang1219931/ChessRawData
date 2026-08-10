import argparse
import os
import re

def get_sort_key(filename):
    match = re.match(re.compile(r"^(.+)-(\d+)\.([a-zA-Z0-9]+)$"), filename)
    if match:
        return (match.group(1), int(match.group(2)))
    return (filename, 0)

def swap_groups_of_eight(directory, dry_run=True):
    pattern = re.compile(r"^(.+)-(\d+)\.([a-zA-Z0-9]+)$")
    
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    print(f"Scanning: {directory}")
    print("Status: DRY RUN (Previewing swapped groups of 8)" if dry_run else "Status: LIVE MODE (Swapping files...)")
    print("-" * 65)

    all_files = [f for f in os.listdir(directory) if pattern.match(f)]
    all_files.sort(key=get_sort_key)

    if not all_files:
        print("No matching files found.")
        return

    # Group files into chunks of 8
    groups = [all_files[i:i + 8] for i in range(0, len(all_files), 8)]
    
    # We will track planned changes: old_path -> temporary_path -> final_path
    rename_plan = []

    for group in groups:
        # Invert the group order to get the swapped targets
        # e.g., [img1, img2, ..., img8] pairs with [img8, img7, ..., img1]
        swapped_group = list(reversed(group))
        
        for original_file, target_file in zip(group, swapped_group):
            orig_match = pattern.match(original_file)
            target_match = pattern.match(target_file)
            
            base_name = orig_match.group(1)
            extension = orig_match.group(3)
            
            # Extract the target number and pad it to 4 digits
            target_num_str = target_match.group(2).zfill(4)
            final_name = f"{base_name}-{target_num_str}.{extension}"
            
            old_path = os.path.join(directory, original_file)
            temp_path = os.path.join(directory, f"TEMP_{original_file}.tmp")
            final_path = os.path.join(directory, final_name)
            
            rename_plan.append({
                'old': old_path,
                'temp': temp_path,
                'final': final_path,
                'print_old': original_file,
                'print_final': final_name
            })

    # Show preview or execute
    for plan in rename_plan:
        if plan['print_old'] != plan['print_final']:
            print(f"Swap: {plan['print_old']} -> {plan['print_final']}")
        else:
            print(f"Unchanged (Middle of group): {plan['print_old']}")

    if not dry_run:
        print("\n[Executing Step 1/2] Moving files to temporary names...")
        for plan in rename_plan:
            try:
                os.rename(plan['old'], plan['temp'])
            except Exception as e:
                print(f"  [ERROR] Temp rename failed for {plan['print_old']}: {e}")

        print("[Executing Step 2/2] Assigning final swapped names...")
        for plan in rename_plan:
            try:
                os.rename(plan['temp'], plan['final'])
            except Exception as e:
                print(f"  [ERROR] Final rename failed for {plan['print_final']}: {e}")

    print("-" * 65)
    print(f"Process complete. Planned operations: {len(rename_plan)}")
    if dry_run:
        print("To execute this swap sequence live, run the command with the --run flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swap file index order in blocks of 8 images.")
    parser.add_argument("folder", help="Path to the target folder containing files.")
    parser.add_argument("--run", action="store_false", dest="dry_run", 
                        help="Execute the swapping process. Without this flag, it only previews changes.")

    args = parser.parse_args()
    swap_groups_of_eight(args.folder, dry_run=args.dry_run)
