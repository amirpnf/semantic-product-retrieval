import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

CSV_PATH = "./data/cleaned_dataset.csv"
MODEL_PATH = "./checkpoints/stylespace_best.pt"
PROJECTION_DIM = 512
BATCH_SIZE = 128  
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "stylespace_products"

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_trained_vision_encoder(model_path, device):

    from model import StyleSpaceModel
    
    print("Loading model structure...")
    model = StyleSpaceModel(projection_dim=PROJECTION_DIM)

    print(f"Loading trained weights from {model_path}...")

    state_dict = torch.load(model_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = load_trained_vision_encoder(MODEL_PATH, device)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded index containing {len(df)} items")

    print(f"Initializing local Qdrant database at: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=PROJECTION_DIM, distance=Distance.COSINE)
    )

    points = []

    df = df.reset_index(drop=True)

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Generating Image Vectors"):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_images = []
        
        batch_rows = []

        for _, row in batch_df.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                tensor = image_transform(img)
                batch_images.append(tensor)
                batch_rows.append(row) 
            except:
                continue
        
        if not batch_images:
            continue

        images_stacked = torch.stack(batch_images).to(device)
        with torch.no_grad():
            vision_outputs = model.vision_encoder(pixel_values=images_stacked)
            images_features = vision_outputs.last_hidden_state[:, 0, :]
            image_embeddings = model.vision_projection(images_features)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            embeddings_list = image_embeddings.cpu().numpy().tolist()

        for step_idx, row in enumerate(batch_rows):
            absolute_id = i + step_idx 

            points.append(
                PointStruct(
                    id=int(absolute_id),
                    vector=embeddings_list[step_idx],
                    payload={
                        "image_path": str(row["image_path"]),
                        "text": str(row["text"])
                    }
                )
            )

        if len(points) >= 1000 or (i + BATCH_SIZE) >= len(df):
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    print(f"Successfully indexed all items into Qdrant collection '{COLLECTION_NAME}' ")


if __name__ == "__main__":
    main()