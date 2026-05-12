import os
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()

class MovieIngestor:
    def __init__(self, csv_path, pinecone_key=None, index_name=None):
        self.csv_path = csv_path
        self.pinecone_key = pinecone_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "movies-index")
        
        if not self.pinecone_key:
            raise ValueError("Missing PINECONE_API_KEY in environment variables.")

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.index = self.pc.Index(self.index_name)

    def preprocess_data(self):
        """Read and clean the movie dataset."""
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        # Select and drop columns based on notebook logic
        cols_to_drop = [
            'budget', 'homepage', 'original_language', 'popularity', 
            'keywords', 'production_companies', 'production_countries', 
            'release_date', 'revenue', 'runtime', 'spoken_languages', 
            'status', 'title', 'vote_average', 'vote_count'
        ]
        
        df_movies = df.drop(columns=cols_to_drop)
        df_movies = df_movies.dropna(axis=0)
        
        # Create combined info for embedding
        df_movies['combined_info'] = df_movies['original_title'].astype(str) + ":" + df_movies['overview'].astype(str)
        
        return df_movies

    def ingest(self, batch_size=100):
        """Embed and upsert data to Pinecone in batches."""
        df_movies = self.preprocess_data()
        
        print(f"Starting ingestion of {len(df_movies)} movies...")
        
        for i in tqdm(range(0, len(df_movies), batch_size), desc="Upserting Batches"):
            batch = df_movies.iloc[i : i + batch_size]
            
            # Generate embeddings
            texts_to_embed = batch['combined_info'].tolist()
            vectors = self.embed_model.embed_documents(texts_to_embed)
            
            # Prepare metadata and IDs
            metadatas = []
            for _, row in batch.iterrows():
                meta = {
                    "title": row['original_title'],
                    "genres": row['genres'],
                    "overview": row['overview'],
                }
                metadatas.append(meta)
            
            ids = batch['id'].astype(str).to_list()
            
            # Create tuples for upsert (id, vector, metadata)
            to_upsert = list(zip(ids, vectors, metadatas))
            
            # Upsert to Pinecone
            self.index.upsert(vectors=to_upsert)
            
        print("Ingestion completed successfully!")

if __name__ == "__main__":
    # Example usage:
    # Make sure to update the path to your tmdb_5000_movies.csv
    CSV_PATH = "data_raw/tmdb_5000_movies.csv" 
    
    try:
        ingestor = MovieIngestor(csv_path=CSV_PATH)
        ingestor.ingest()
    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
