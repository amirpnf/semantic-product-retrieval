# semantic-product-retrieval


### Overview

A real-time multimodal search engine, leveraging a dual-encoder CLIP architecture. This system allows users to search products seamlessly using text queries, images, or a combination of both, powered by PyTorch, Qdrant vector database, and FastAPI.

### How it works ?

The core of the search engine relies on a dual-encoder architecture powered by CLIP (Contrastive Language-Image Pre-training). This model projects both text queries and product images into a shared, high-dimensional vector space where semantic similarity corresponds to geometric proximity.

1. Unified Embedding Space

CLIP consists of two separate neural networks trained jointly:

    Text Encoder: Converts textual descriptions, titles, or search terms into dense vector embeddings.

    Image Encoder: Converts product images (photos, banners, graphics) into dense vector embeddings of the exact same dimensionality.

Since both modalities share the same latent space, a text query like "red running shoes" can be directly compared against product images, and vice versa, without needing complex cross-modal translation layers.

2. Data Ingestion & Indexing

- Processing : Product catalogs containing titles, descriptions, and images are passed through the PyTorch dual-encoder models to generate vector representations.

- Storage : These vectors, along with essential metadata (product ID, title, price, image URL), are indexed and stored in Qdrant. Qdrant uses advanced Approximate Nearest Neighbor (ANN) algorithms (such as HNSW) to make sub-millisecond retrieval possible across millions of records.

3. Search Workflow

When a user executes a search through the FastAPI backend:

1. Query Encoding: The input is sent to the appropriate CLIP encoder to instantly generate a query vector.

2. Vector Similarity Search: The query vector is dispatched to Qdrant, which performs a cosine similarity or dot-product search against the pre-computed product vectors.

3. Retrieval & Ranking: Qdrant returns the top-k nearest neighbors ranked by semantic relevance.

4. API Response: FastAPI wraps the retrieved product metadata and returns structured JSON results to the client.


### Quickstart

