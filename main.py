from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# Load the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Root endpoint
@app.get("/")
def root():
    return {"message": "MindCare API running"}

# Define input data structure for POST
class InputText(BaseModel):
    text: str

# Sentiment analysis endpoint
@app.post("/infer")
def infer(data: InputText):
    result = sentiment_model(data.text)
    return {"input": data.text, "prediction": result}
