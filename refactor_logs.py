import os
import shutil

def should_delete_due_to_checkpoint(directory):
    """Check if the given directory has a 'checkpoint' subdirectory with fewer than 3 files."""
    checkpoint_path = os.path.join(directory, "checkpoint")
    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        num_files = len([name for name in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, name))])
        return num_files < 3
    return False

def should_delete_due_to_missing_checkpoint(parent_dir, child_dir):
    """Check if the great-grandparent directory is named 'logs', its parent is '.', and the child directory doesn't have a 'checkpoint' subdirectory."""
    grandparent_dir = os.path.dirname(parent_dir)
    great_grandparent_dir = os.path.dirname(grandparent_dir)
    great_great_grandparent_dir = os.path.dirname(great_grandparent_dir)
    
    if os.path.basename(great_grandparent_dir) == "logs" and great_great_grandparent_dir == ".":
        checkpoint_path = os.path.join(parent_dir, child_dir, "checkpoint")
        return not os.path.exists(checkpoint_path)
    return False

def should_delete_if_empty(directory):
    """Check if an ancestor of a directory is named 'logs' and the directory itself is empty."""
    
    current_dir = directory
    while True:
        parent_dir = os.path.dirname(current_dir)
        
        # If we reach the root directory without finding "logs", exit
        if parent_dir == current_dir:
            break
            
        # If we find a parent named "logs", check if the directory is empty
        if os.path.basename(parent_dir) == "logs":
            return is_directory_empty(directory)
        
        current_dir = parent_dir

    return False


def is_directory_empty(directory):
    """Check if a directory is empty."""
    return not bool(os.listdir(directory))

def main():
    start_dir = "."  # Start from the current directory
    for root, dirs, files in os.walk(start_dir, topdown=False):
        for directory in dirs:
            full_path = os.path.join(root, directory)
            if (should_delete_due_to_checkpoint(full_path) or 
                should_delete_due_to_missing_checkpoint(root, directory) or 
                should_delete_if_empty(full_path)):
                print(f"Deleting directory: {full_path}")
                shutil.rmtree(full_path)

                # Continuously check parent directories and remove them if they become empty.
                current_check_dir = root
                while current_check_dir != start_dir and is_directory_empty(current_check_dir):  # Making sure we don't move beyond our start directory.
                    print(f"Deleting empty directory: {current_check_dir}")
                    parent_check_dir = os.path.dirname(current_check_dir)
                    shutil.rmtree(current_check_dir)
                    current_check_dir = parent_check_dir

if __name__ == "__main__":
    main()
