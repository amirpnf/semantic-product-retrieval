import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import torch
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from src.models.model import StyleSpaceModel
from PIL import Image

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "stylespace_products"
MODEL_PATH = "./checkpoints/stylespace_best.pt"
PROJECTION_DIM = 512

st.set_page_config(page_title="StyleSpace Search Engine", layout="wide")
st.title("🛍️ StyleSpace: Multimodal Product Search")
st.write("Search your retail collection semantically utilizing your fine-tuned dual-encoder network.")

@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = StyleSpaceModel(projection_dim=PROJECTION_DIM)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    client = QdrantClient(path=QDRANT_PATH)
    return tokenizer, model, client, device

tokenizer, model, client, device = load_resources()

query = st.text_input("What are you looking for today? (e.g., 'vintage leather boots', 'floral casual summer dress')")
top_k = st.slider("Number of match results to show", min_value=1, max_value=24, value=8)

if query:
    with st.spinner("Searching the catalog..."):
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_outputs = model.text_encoder(**inputs)
            text_features = text_outputs.last_hidden_state[:, 0, :]
            text_embeddings = model.text_projection(text_features)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            query_vector = text_embeddings.squeeze(0).cpu().numpy().tolist()
            
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        results = response.points 
        
        if results:
            cols = st.columns(4)  
            for idx, hit in enumerate(results):
                col_idx = idx % 4
                with cols[col_idx]:
                    payload = hit.payload
                    score = hit.score
                    
                    try:
                        img = Image.open(payload["image_path"])
                        st.image(img, use_container_width=True)
                        st.markdown(f"**Similarity Score:** `{score:.3f}`")
                        st.caption(f"*Metadata:* {payload['text']}")
                    except Exception as e:
                        st.error(f"Image load failure: {payload['image_path']}")
        else:
            st.info("No semantic matches discovered for your phrase.")