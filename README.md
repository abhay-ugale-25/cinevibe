# CineVibe

Finding a movie to watch is usually a frustrating experience. You might be in the mood for something specific—like "a creepy space movie where nobody trusts each other"—but streaming platforms force you to click through generic categories like "Sci-Fi" or "Thriller."

I built CineVibe to solve that exact problem. It is a decoupled backend microservice that ditches rigid tagging systems. Instead, it lets users search using natural language "vibes," leveraging vector similarity and generative AI to return highly personalized recommendations.

### How the Engine Operates

The architecture is split into two distinct parts to keep the live API incredibly fast.

* **Offline Data Processing:** Before the API even runs, an ingestion script processes a raw dataset of thousands of movies. It uses HuggingFace (`all-MiniLM-L6-v2`) to turn movie overviews and metadata into mathematical vectors, which are then securely upserted into a Pinecone vector database.
* **The Live API Pipeline:** When a request hits the FastAPI server, the engine instantly converts the user's text query into a vector. Pinecone runs a similarity search to grab the most contextually relevant films. Finally, Google's Gemini 2.5 Flash model takes those matches and synthesizes a conversational response, explaining to the user exactly why the top pick fits their specific mood.

### The Tech Stack

* **API Framework:** FastAPI
* **Vector Storage:** Pinecone
* **LLM Synthesis:** Google Gemini 2.5 Flash
* **Embedding Model:** HuggingFace sentence-transformers
* **Language:** Python 3

### Hitting the Endpoint

The service exposes a single, clean REST endpoint that any front-end application can plug into.

**POST** `/api/v1/recommend`

**Payload:**

```json
{
  "query": "I want a dark, psychological thriller with a massive plot twist"
}

```

**Response:**

```json
{
  "query": "I want a dark, psychological thriller with a massive plot twist",
  "recommendation": "Here is the absolute best match for your vibe...\n\n### Shutter Island\nThis film drops you onto a remote island psychiatric facility... [Response continues with reasoned explanation and alternatives]"
}

```

### Getting it Running Locally

If you want to spin this up on your own machine, follow these steps.

First, create a `.env` file in the root directory and drop in your credentials:

```env
PINECONE_API_KEY=your_actual_key
PINECONE_INDEX_NAME=movies-index
GOOGLE_API_KEY=your_actual_key

```

Next, grab the dependencies:

```bash
pip install -r requirements.txt

```

Finally, start the local server:

```bash
uvicorn api:app --reload

```

You can navigate right to `http://127.0.0.1:8000/docs` to interact with the API through the automatic Swagger interface.
