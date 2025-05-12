from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import openai
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("study-assistant")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Study Assistant API", version="1.0.0")

# Sample knowledge base
educational_passages = [
    "سلول‌های گیاهی دارای دیواره سلولی و کلروپلاست هستند و در فتوسنتز نقش دارند.",
    "جانوران مهره‌دار شامل پستانداران، پرندگان، خزندگان، دوزیستان و ماهی‌ها هستند.",
    "فرایند فتوسنتز در کلروپلاست‌های سلول‌های گیاهی انجام می‌شود.",
    "پادتن‌ها توسط سیستم ایمنی بدن برای مقابله با عوامل بیماری‌زا تولید می‌شوند."
]

# Request schema
class QuestionRequest(BaseModel):
    question: str = Field(..., example="سلول گیاهی چیست؟")

# Response schema
class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_passage: str

# NLP Utility: Get embedding from OpenAI
def get_embedding(text: str) -> List[float]:
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error retrieving embedding: {e}")
        raise

# NLP Utility: Find most relevant passage
def find_most_similar_passage(question: str, passages: List[str]) -> str:
    try:
        question_embedding = get_embedding(question)
        passage_embeddings = [get_embedding(p) for p in passages]
        similarities = cosine_similarity([question_embedding], passage_embeddings)[0]
        most_similar_index = int(np.argmax(similarities))
        return passages[most_similar_index]
    except Exception as e:
        logger.error(f"Error finding similar text: {e}")
        raise

# LLM Utility: Generate answer with OpenAI ChatCompletion
def generate_answer(question: str, context: str) -> str:
    try:
        prompt = f"Question: {question}\nEducational text: {context}\nAnswer:"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an educational assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error generating the response: {e}")
        raise

# Main API endpoint
@app.post("/ask", response_model=AnswerResponse, summary="Question to the learning assistant")
def ask_question(req: QuestionRequest):
    try:
        logger.info(f"Question received: {req.question}")
        best_passage = find_most_similar_passage(req.question, educational_passages)
        answer = generate_answer(req.question, best_passage)
        logger.info("Response generated successfully")
        return AnswerResponse(
            question=req.question,
            answer=answer,
            source_passage=best_passage
        )
    except Exception as e:
        logger.exception("Error processing the request")
        raise HTTPException(status_code=500, detail="Error processing the question: " + str(e))
