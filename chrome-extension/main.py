from fastapi import FastAPI, HTTPException
import httpx
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://news.naver.com", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 가져오기
model_name = 'yeti-s/kobart-title-generator'
title_generator = pipeline("text2text-generation", model=model_name)

@app.post("/scrape")
async def scrape(data: dict):    
    title = data.get("title")
    content = data.get("content")
    
    # 모델을 사용하여 제목 생성
    generated_titles = title_generator(f"{content}", max_length=50, min_length=10, num_return_sequences=1)
    generated_title = generated_titles[0]['generated_text'].strip()

    return {"generated_title": generated_title}