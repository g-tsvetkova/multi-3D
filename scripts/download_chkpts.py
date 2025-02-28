import gdown
import os

def download_checkpoints():
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Google Drive folder ID (extract from your folder URL)
    folder_url = "YOUR_FOLDER_URL"
    
    # List of checkpoint files to download
    checkpoints = [
        "checkpoint.pth",
        "checkpoint_20k.pth",
        "checkpoint_40k.pth",
        "checkpoint_60k.pth"
    ]
    
    for checkpoint in checkpoints:
        output_path = f"checkpoints/{checkpoint}"
        if not os.path.exists(output_path):
            print(f"Downloading {checkpoint}...")
            # You'll need to replace these with actual file IDs from Google Drive
            file_id = "YOUR_FILE_ID_FOR_" + checkpoint
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        else:
            print(f"{checkpoint} already exists, skipping...")

if __name__ == "__main__":
    download_checkpoints()