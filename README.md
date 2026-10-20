# Semantic Product Retrieval

A real-time multimodal e-commerce search engine powered by a custom dual-encoder architecture. Built with PyTorch, Qdrant vector database, and FastAPI, this system maps product images and text descriptions into a shared semantic space, enabling cross-modal retrieval (such as searching for products using natural language text queries or reference images).

## Architecture & How It Works

The core of the search engine relies on a custom dual-encoder architecture trained with a symmetric contrastive loss. This model combines a Vision Transformer (`ViT`) for visual feature extraction and a Transformer language model (`DistilBERT`) for text understanding, projecting both modalities into a shared high-dimensional vector space.

1. **Unified Embedding Space**
   * **Text Encoder (`DistilBERT`):** Converts product titles, descriptions, and user search terms into dense vector embeddings.
   * **Vision Encoder (`ViT`):** Converts product images into dense vector embeddings of the exact same dimensionality.
   * Because both modalities share the same latent space, a text query like *"red running shoes"* can be directly compared against stored product image embeddings without needing complex cross-modal translation layers.

2. **Data Ingestion & Indexing**
   * **Data Pipeline:** Product metadata and images are parsed from the Amazon Berkeley Objects (ABO) dataset via `prepare_data.py`.
   * **Storage & Retrieval:** Product images are passed through the trained PyTorch vision encoder to generate normalized vector representations. These vectors, along with their textual metadata, are stored and indexed in **Qdrant** using Approximate Nearest Neighbor (ANN) search (HNSW) for sub-millisecond retrieval.

3. **Search Workflow**
   1. **Query Encoding:** The text or image input is passed through the respective encoder via the FastAPI backend to generate a query vector.
   2. **Vector Similarity Search:** The query vector is queried against Qdrant using cosine similarity.
   3. **Retrieval & Ranking:** Qdrant returns the top-$k$ nearest neighbors ranked by semantic relevance.

---

## Project Structure

```text
├── checkpoints/         # Saved model weights (.pt files)
├── data/                # Raw datasets, extracted files, and cleaned CSVs
├── qdrant_db/           # Local Qdrant vector database storage
├── src/
│   ├── api/             # FastAPI application and route handlers
│   ├── data/            # Dataset definitions, dataloaders, and ABO download scripts
│   ├── models/          # PyTorch model definition (ViT + DistilBERT) and Contrastive Loss
│   └── pipelines/       # Training and Qdrant vector indexing scripts
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
---

## Tech Stack

- **ML/Deep Learning** : PyTorch, Transformers(Hugging Face), ViT, DistilBERT

- **Vector Database** : Qdrant

- **Backend API** : FastAPI, Uvicorn

- **Data Processing** : Pandans, Numpy, Pillow, Torchvision

---

## Getting Started

#### Prerequisites

- **Python 3.10+**
- **CUDA-compatible GPU (recommended for training)**

#### Installation

1. **Clone the repository** : 
```bash
git clone [https://github.com/your-username/semantic-product-retrieval.git](https://github.com/your-username/semantic-product-retrieval.git)
cd semantic-product-retrieval
```


2. **Create and activate a virtual environment** : 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies** : 

```bash
pip install -r requirements.txt
```

--

## Usage Pipeline

Make sure to run Python commands with the proper PYTHONPATH so modules can locate the src package.

1. **Download and Prepare Data** : 

```bash
python src/data/download_abo.py
python src/data/prepare_data.py
```

2. **Train the Dual-Encoder Model** : 
```bash
PYTHONPATH=. python src/pipelines/train.py
```

3. **Index Product Vectors into Qdrant** : 
```bash
PYTHONPATH=. python src/pipelines/index_vector.py
```

4. **Run the FastAPI Backend** : 
```bash
PYTHONPATH=. uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

