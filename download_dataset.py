import kagglehub
import shutil
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the dataset (downloads to a temp dir)
path = kagglehub.dataset_download("msambare/fer2013")

# Move all files from kagglehub temp path to ./data
for item in os.listdir(path):
    src_path = os.path.join(path, item)
    dst_path = os.path.join("data", item)

    # If it's a folder (e.g., images organized by class), copy entire directory
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src_path, dst_path)

print("✅ Dataset copied to: ./data")
