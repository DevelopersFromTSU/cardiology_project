import os
import re
import math
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / "pipeline" / ".env"
if not env_path.exists():
    env_path = BASE_DIR / "pipeline" / ".env.txt"

if env_path.exists():
    # Открываем с кодировкой utf-8-sig, которая автоматически уничтожает невидимый BOM-символ Блокнота
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            # Вырезаем артефакты вроде "", если они случайно скопировались в файл
            line = re.sub(r"^\\s*", "", line)
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")  # Очищаем кавычки вокруг значений
                os.environ[key] = val

print(f"📁 Файл окружения: '{env_path}' (Существует? {env_path.exists()})")
print(f"🔑 Проверка памяти: FOLDER_ID='{os.getenv('YANDEX_FOLDER_ID')}' | API_KEY='{os.getenv('YANDEX_API_KEY')[:10] if os.getenv('YANDEX_API_KEY') else None}...'")

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
    Превращает разговорную речь пациента в строгий медицинский поисковый запрос
    через YandexGPT API (Foundation Models).
    """
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    folder_id = os.getenv("YANDEX_FOLDER_ID")
    api_key = os.getenv("YANDEX_API_KEY")

    if not folder_id or not api_key:
        print(
            "⚠️ Учетные данные Yandex (YANDEX_FOLDER_ID или YANDEX_API_KEY) не найдены. Поиск пойдет по сырому запросу.")
        return patient_text

    # Для статических API-ключей в Яндекс.Облаке используется схема Api-Key, а не Bearer
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json"
    }

    sys_instr = (
        "Ты — специализированный поисковый процессор (NER & Query Optimizer) для медицинской базы знаний RAG. "
        "Твоя ЕДИНСТВЕННАЯ задача — преобразовать разговорный запрос или жалобу пациента в плотный поисковый вектор "
        "из стандартизированных клинических терминов и тегов.\n\n"
        "ЖЕСТКИЕ ПРАВИЛА ГЕНЕРАЦИИ:\n"
        "1. ТЕЛЕГРАФНЫЙ СТИЛЬ (БЕЗ ВОДЫ): КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать связные предложения, вопросы ('какое должно быть', 'подскажите'), "
        "глаголы действия или вводные конструкции. Исключи все бытовые стоп-слова. Выводи ТОЛЬКО ключевые клинические сущности через пробел.\n"
        "2. СТАНДАРТИЗАЦИЯ И АББРЕВИАТУРЫ: Переводи любые разговорные симптомы и названия болезней в официальные медицинские термины, добавляя аббревиатуру в скобках. "
        "Примеры: 'гипертония' -> 'артериальная гипертензия (АГ)', 'сахар' -> 'сахарный диабет (СД)'.\n"
        "3. УНИВЕРСАЛЬНЫЕ КОЛИЧЕСТВЕННЫЕ ПОРОГИ: Если в запросе есть числовые показатели (возраст, АД, пульс), сохраняй точную цифру И формируй из нее логический порог "
        "(например: 'возраст 82 года' -> 'возраст старше 80 лет >= 80 лет').\n"
        "4. ФИКСАЦИЯ ЛЕКАРСТВ И ПРОЦЕДУР (КРИТИЧЕСКИ ВАЖНО): Если пациент упоминает конкретные препараты, классы лекарств (например: диуретики, статины, таблетки от давления) "
        "или высказывает страхи по их поводу — ОБЯЗАТЕЛЬНО сохраняй их названия в запросе. Добавляй к ним теги: 'противопоказания побочные эффекты использовать с осторожностью'.\n"
        "5. РАСШИРЕННЫЙ КЛИНИЧЕСКИЙ ИНТЕНТ (ДИАГНОЗ + ЛЕЧЕНИЕ): Чтобы база нашла не только нормы, но и что делать, всегда добавляй двойной набор тегов:\n"
        "   - Для оценки ситуации: 'целевые значения целевой уровень классификация риск SCORE2'.\n"
        "   - Для действий врача: 'лечение стратегия лекарственной терапии алгоритм стартовая терапия шаг 1 шаг 2'.\n"
        "6. ФОРМАТ ВЫДАЧИ: Выводи ТОЛЬКО итоговую строку тегов без кавычек, точек и пояснений."
    )

    payload = {
        "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.1,
            "maxTokens": "300"
        },
        "messages": [
            {"role": "system", "text": sys_instr},
            {"role": "user", "text": patient_text}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        # В структуре ответа Yandex Cloud ML текст лежит в result -> alternatives -> message -> text
        response_data = response.json()
        refined_query = response_data['result']['alternatives'][0]['message']['text'].strip()
        return refined_query

    except Exception as e:
        print(f"⚠️ Ошибка перефразирования через YandexGPT: {e}")
        # В случае сбоя возвращаем оригинальный текст, чтобы конвейер не падал
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

        # Вывод в консоль для отладки можно оставить
        print(f"[⭐ Точность: {percentage_score}% | Страница: {page}] -> {text}")

        # [НОВОЕ]: Реальное добавление данных в массив
        retrieved_texts.append({
            "text": text,
            "page": page,
            "score": percentage_score
        })

    return retrieved_texts


if __name__ == "__main__":
    # 1. Прописываем живой вопрос пациента
    patient_question = ("У моего дедушки 82 года, гипертония в сочетании с фибрилляцией предсердий. Пульс постоянно "
                        "высокий, около 88 ударов в минуту. Стартовая терапия не помогла снизить давление до нормы. "
                        "Какое целевое верхнее давление ему нужно держать и какие препараты нужно добавить на втором "
                        "шаге лечения?")
    print(f"👤 Вопрос пациента: {patient_question}\n")

    # 2. Переводим его в медицинские термины через Яндекс
    print("🤖 Отправляем запрос в YandexGPT для перефразирования...")
    medical_query = rewrite_patient_query(patient_question)
    print(f"🔍 Сформированный медицинский запрос: {medical_query}\n")

    # 3. Ищем в базе Qdrant и реранжируем результаты
    print("🚀 Поиск по гибридной базе знаний BGE-M3...")
    hybrid_search(search_query=medical_query, top_k=5)

# формула сигмоиды для обозначения итоговых цифр точности в диапазоне от 0 до 100 процентов
