import os
import re
import requests
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from qdrant_client import models
from FlagEmbedding import BGEM3FlagModel
import math

load_dotenv()

qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_docs")

print("⏳ Загрузка мощной гибридной модели BGE-M3...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("⏳ Загрузка модели реранкера (Cross-Encoder)...")
reranker = CrossEncoder('BAAI/bge-reranker-base')
print("✅ Все модели готовы!")

def logit_to_percentage(score: float) -> float:
    """
    НОВАЯ ФУНКЦИЯ: Преобразует сырой логит реранкера в проценты от 0 до 100
    через математическую функцию сигмоиды.
    """
    probability = 1 / (1 + math.exp(-score))
    return round(probability * 100, 2)

def rewrite_patient_query(patient_text: str) -> str:
    """
    Превращает разговорную речь пациента в строгий медицинский поисковый запрос через YandexGPT.
    """
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    api_key = os.getenv("YANDEX_API_KEY")

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json"
    }

    data = {
        "modelUri": f"gpt://{folder_id}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 2000
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты — опытный врач-кардиолог. Переведи бытовые жалобы пациента в один лаконичный "
                    "поисковый запрос из ключевых медицинских терминов для поиска в клинических рекомендациях.\n\n"
                    "Правила:\n"
                    "1. Не пиши списки, анкеты и знаки двоеточия.\n"
                    "2. Переводи числовой возраст в медицинскую категорию (например: 35 лет -> 'у молодых пациентов' или 'взрослые'; 75 лет -> 'в пожилом возрасте').\n"
                    "3. Выдавай результат одной строкой.\n\n"
                    "Пример:\n"
                    "Пациент: 'Мне 25, дома намерил 135 на 85, болит затылок'\n"
                    "Результат: 'высокое нормальное артериальное давление симптомы цефалгии у молодых пациентов'"
                )
            },
            {
                "role": "user",
                "text": patient_text
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:
        print(f"Ошибка перефразирования: {e}")
    return patient_text


def hybrid_search(search_query: str, top_k: int = 5):
    # Генерируем оба вектора для поискового запроса
    outputs = model.encode([search_query], return_dense=True, return_sparse=True)
    dense_query = outputs['dense_vecs'][0].tolist()
    sparse_dict = outputs['lexical_weights'][0]

    sparse_query = models.SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()]
    )

    # Делаем один нативный гибридный запрос через prefetch и RRF (Reciprocal Rank Fusion)
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=40),
            models.Prefetch(query=sparse_query, using="sparse", limit=40)
        ],
        query=models.RrfQuery(rrf=models.Rrf()),
        limit=top_k * 3,  # Берем с запасом для последующего реранкера
        with_payload=True
    )

    candidate_list = response.points

    # Реранжирование результатов
    if candidate_list:
        pairs = [[search_query, hit.payload.get('text', '')] for hit in candidate_list]
        rerank_scores = reranker.predict(pairs)
        scored_candidates = sorted(zip(candidate_list, rerank_scores), key=lambda x: x[1], reverse=True)
        final_points = scored_candidates[:top_k]
    else:
        final_points = []

    print(f"\nВыдача результатов (Возвращено: {len(final_points)})\n" + "=" * 50)
    retrieved_texts = []
    for i, (hit, score) in enumerate(final_points, 1):
        text = hit.payload.get('text', '')
        page = hit.payload.get('page', 'Неизвестно')
        percentage_score = logit_to_percentage(score)
        print(f"[⭐ Точность: {percentage_score}% | Страница: {page}] -> {text}")

    return retrieved_texts


if __name__ == "__main__":
    # 1. Прописываем живой вопрос пациента
    patient_question = "Мне 35 лет. Стал часто измерять давление дома, верхнее стабильно держится в районе 135, а нижнее около 87. Голова при этом немного тяжелая. Подскажите, это уже считается болезнью или еще нормально?"
    print(f"👤 Вопрос пациента: {patient_question}\n")

    # 2. Переводим его в медицинские термины через Яндекс
    print("🤖 Отправляем запрос в YandexGPT для перефразирования...")
    medical_query = rewrite_patient_query(patient_question)
    print(f"🔍 Сформированный медицинский запрос: {medical_query}\n")

    # 3. Ищем в базе Qdrant и реранжируем результаты
    print("🚀 Поиск по гибридной базе знаний BGE-M3...")
    hybrid_search(search_query=medical_query, top_k=5)

# формула сигмоиды для обозначения итоговых цифр точности в диапазоне от 0 до 100 процентов
