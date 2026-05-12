import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configure logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class CineVibeEngine:
    def __init__(self):
        print("Initializing CineVibe Engine...")

        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if not self.pinecone_key or not self.google_key:
            raise ValueError("Missing API keys. Please provide them in .env or as arguments.")

        # Embedding model using updated langchain_huggingface
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Default to cpu for broader compatibility in production
        )
        print("Embedding model loaded!")

        # Pinecone connection
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone Index: {self.index_name}")

        # Gemini AI setup
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash', 
            google_api_key=self.google_key,
            temperature=0.6
        )
        print("Gemini AI loaded!")

    def search(self, query, top_k=5):
        """Perform similarity search in Pinecone."""
        query_vector = self.embed_model.embed_query(query)
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return results

    def format_context(self, results):
        """Format Pinecone results into a readable string for the LLM."""
        context_string = ""
        for idx, match in enumerate(results['matches'], start=1):
            meta = match.get('metadata', {})
            title = meta.get('title', 'Unknown Title')
            overview = meta.get('overview', 'No overview available')
            genres = meta.get('genres', 'Unknown Genres')

            context_string += f"{idx}. Title: {title}\n"
            context_string += f"   Genres: {genres}\n"
            context_string += f"   Overview: {overview}\n\n"
        return context_string

    def create_prompt(self, user_query, context):
        """Create the prompt for Gemini."""
        prompt = f"""
        You are a movie expert. The user wants a movie with this vibe: "{user_query}".

        Here are the top 5 matches from our database:
        {context}

        Task:
        1. Pick the ONE best movie from this list.
        2. Explain why it fits the user's mood creatively.
        3. Suggest 2 alternatives.
        4. Use emojis.
        5. Align the output in an attractive way.
        """
        return prompt

    def recommend(self, query):
        """Full pipeline: search -> context -> prompt -> LLM recommendation."""
        if not query:
            return "Describe your vibe!"

        search_result = self.search(query)
        context = self.format_context(search_result)
        prompt = self.create_prompt(query, context)
        response = self.llm.invoke(prompt)

        return response.content
