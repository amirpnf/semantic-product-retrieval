import pandas as pd
import gzip
import json
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path("./data")
LISTINGS_DIR = DATA_DIR / "listings" / "metadata"
IMAGES_META_PATH = DATA_DIR / "images" / "metadata" / "images.csv.gz"
IMAGES_DIR = DATA_DIR / "images" / "small"
OUTPUT_CSV = DATA_DIR / "cleaned_dataset.csv"

def extract_english_text(item_name_list):

    if not isinstance(item_name_list, list):
        return None
    
    for entry in item_name_list:
        if entry.get("language_tag") in ["en_US", "en_IN", "en_GB"]:
            return entry.get("value")
        
    return None

def main():

    print("Step 1 : parsing product metadata...")

    json_files = list(LISTINGS_DIR.glob("listings_*.json.gz"))
    parsed_items = []

    for file_path in tqdm(json_files, desc="parsing JSON metadata"):

        with gzip.open(file_path, 'rt', encoding='utf-8') as f:

            for line in f:
                data = json.loads(line)
                item_id = data.get("item_id")
                text = extract_english_text(data.get("item_name"))
                main_image_id = data.get("main_image_id")

                if item_id and text and main_image_id:
                    parsed_items.append({
                        "item_id" : item_id,
                        "text" : text.strip(),
                        "image_id" : main_image_id
                    })

    df_text = pd.DataFrame(parsed_items)
    
    print(f"Parsed {len(df_text)} products with valid English names.")
    print("Step 2 : Mapping to actual image file paths...")

    df_images = pd.read_csv(IMAGES_META_PATH)
    df_merged = pd.merge(df_text, df_images, on="image_id", how="inner")

    df_final = pd.DataFrame({
        "image_path" : df_merged["path"].apply(lambda p: str(IMAGES_DIR / p)),
        "text" : df_merged["text"]
    })

    print("Step 3: Stripping missing files and duplicates...")
    df_final["exists"] = df_final["image_path"].apply(lambda p: Path(p).exists())
    df_final = df_final[df_final["exists"] == True].drop(columns=["exists"])
    
    df_final = df_final.dropna(subset=["text"])
    df_final = df_final.drop_duplicates(subset=["image_path"])

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Cleaned dataset index saved to: {OUTPUT_CSV}")
    print(f"Total valid image-text pairs ready: {len(df_final)}")

if __name__ == "__main__":
    main()


