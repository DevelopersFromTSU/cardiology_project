import os
import json
import logging
import asyncio
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

# ------------------- Настройки -------------------
GEMINI_API_KEY = "AIzaSyDI32gW0JbKy1ciAjMfiQC1qaDt-FPU8Mg"
MODEL_NAME = "gemini-2.5-flash"
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# Папки для истории и логов
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Настройка логирования
logging.basicConfig(
    filename=LOGS_DIR / "chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Монтируем статику (для index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------- Работа с историей -------------------
def load_history(session_id: str) -> List[dict]:
    file = HISTORY_DIR / f"{session_id}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    return []

def save_history(session_id: str, history: List[dict]):
    file = HISTORY_DIR / f"{session_id}.json"
    file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------- Заглушка RAG-поиска -------------------
def search_similar(query: str, top_k: int = 3) -> List[str]:
    """
    Здесь вы вызываете свою функцию поиска по векторной базе.
    Пока возвращает демо-чанки.
    """
    # Замените на реальный поиск
    return [
        "Чанк 1: Что-то про вуз...",
        "Чанк 2: Расписание на четверг...",
        "Чанк 3: Помощь студентам..."
    ][:top_k]

# ------------------- Генерация ответа с Gemini и потоком -------------------
async def generate_stream_response(session_id: str, user_msg: str):
    chunks = search_similar(user_msg)
    context = "\n".join(chunks)

    prompt = f"""Ты — полезный ассистент для сотрудников вуза. Используй предоставленный контекст.
Контекст:
{context}

Вопрос пользователя: {user_msg}
Ответ:"""

    history = load_history(session_id)

    # Формируем сообщения для модели
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    full_answer = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        ):
            text = chunk.text
            if text:
                full_answer += text
                yield f"data: {json.dumps({'token': text})}\n\n"
            await asyncio.sleep(0)
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        yield f"data: {json.dumps({'token': f'Ошибка генерации: {str(e)}'})}\n\n"

    yield "data: [DONE]\n\n"

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": full_answer})
    save_history(session_id, history)
    logging.info(f"Session: {session_id} | User: {user_msg} | Bot: {full_answer[:200]}...")

# ------------------- Эндпоинты -------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    if not message:
        return {"error": "No message"}

    return StreamingResponse(
        generate_stream_response(session_id, message),
        media_type="text/event-stream"
    )

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return load_history(session_id)

@app.get("/sessions")
async def list_sessions():
    files = HISTORY_DIR.glob("*.json")
    sessions = [f.stem for f in files]
    return {"sessions": sessions}

@app.get("/logs")
async def get_logs():
    log_file = LOGS_DIR / "chat.log"
    if log_file.exists():
        return FileResponse(log_file, media_type="text/plain")
    return {"error": "No logs yet"}

@app.get("/")
async def root():
    return FileResponse("static/index.html")