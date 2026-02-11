# CineVibe: Semantic Movie Search Engine

**CineVibe** is a Semantic Search Engine that allows users to find movies based on "vibes," emotions, and complex plot descriptions rather than just keywords.

> *"I want a movie that feels like a warm hug on a rainy day."*

## Key Features
- **Semantic Understanding:** Uses **Vector Embeddings** (`all-MiniLM-L6-v2`) to capture the *meaning* of a query.
- **RAG Architecture:** Retrieval-Augmented Generation pipeline using **Pinecone** (Vector DB) and **Google Gemini 2.5 Flash** (LLM).
- **Hybrid Search:** Combines semantic similarity with metadata filtering (Title, Overview).
- **Microservice Design:** Encapsulated logic in a reusable `CineVibeEngine` class.

## Tech Stack
- **LLM:** Google Gemini 2.5 Flash
- **Vector Database:** Pinecone (Serverless)
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Orchestration:** LangChain
- **Language:** Python 3.10+

## How It Works
1. **Ingestion (ETL):** The system processes the TMDB 5000 Movie Dataset, cleaning and combining features (Title + Overview).
2. **Embedding:** Text is converted into 384-dimensional vectors using a local Transformer model.
3. **Storage:** Vectors are upserted to a Pinecone index with metadata.
4. **Inference:**
   - User Query → Vector Embedding
   - Vector Search (Top-K) → Retrieved Context
   - Context + Prompt → LLM Recommendation

## Dataset
[Kaggle TMDB 5000 movies dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download&select=tmdb_5000_movies.csv)

## How to Run
1. Open `cinevibe.ipynb` in **Google Colab**.
2. Add your API keys to the Colab Secrets Manager:
   - `pinecone`: Your Pinecone API Key
   - `GoogleAIStudio`: Your Google Gemini API Key
3. Run the notebook cells to initialize the engine and start the interactive CLI.

---
*Built as a Semantic Search Engineering Sprint.*