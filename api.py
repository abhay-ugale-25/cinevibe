from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.engine import CineVibeEngine
import uvicorn

# Define request model
class RecommendationRequest(BaseModel):
    query: str

# Initialize FastAPI app
app = FastAPI(title="CineVibe API", description="AI-powered movie recommendation microservice")

# Initialize the Recommendation Engine
# The engine will load API keys from .env automatically
try:
    engine = CineVibeEngine()
except Exception as e:
    print(f"Failed to initialize CineVibeEngine: {e}")
    engine = None

@app.get("/")
async def root():
    return {"message": "Welcome to CineVibe API! Use /api/v1/recommend to get movie suggestions."}

@app.post("/api/v1/recommend")
async def recommend(request: RecommendationRequest):
    """
    Accepts a user's 'vibe' query and returns a synthesized movie recommendation.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Recommendation engine is not initialized. Check server logs.")

    try:
        recommendation = engine.recommend(request.query)
        return {
            "query": request.query,
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during recommendation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
