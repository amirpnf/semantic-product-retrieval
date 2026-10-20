import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("./data")
URLS = {
    "abo-listings.tar": "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar",
    "abo-images-small.tar": "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"
}

class DownloadProgressBar(tqdm):
    """Generates a visual progress bar for download feedback"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_tar(tar_path, extract_path):
    print(f"Extracting {tar_path.name} to {extract_path}...")
    with tarfile.open(tar_path, "r:") as tar:
        tar.extractall(path=extract_path)
    print(f"Finished extracting {tar_path.name}.")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url in URLS.items():
        archive_path = DATA_DIR / filename
        
        if not archive_path.exists():
            print(f"\nStarting download: {filename}")
            try:
                download_url(url, archive_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue
        else:
            print(f"\n{filename} already exists. Skipping download.")

        folder_name = filename.replace(".tar", "")
        extraction_indicator = DATA_DIR / folder_name
        
        if not extraction_indicator.exists():
            extract_tar(archive_path, DATA_DIR)
            print(f"Cleaning up {filename} to reclaim space...")
            archive_path.unlink()
        else:
            print(f"{folder_name} already extracted. Skipping extraction.")

    print("\n Dataset setup complete! Your './data' folder is ready.")

if __name__ == "__main__":
    main()