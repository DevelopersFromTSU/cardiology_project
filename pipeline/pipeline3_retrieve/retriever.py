import os
import re
import math
import requests
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

# 1. Поиск .env в корне проекта
BASE_DIR = Path(__file__).resolve().parent
root_env = BASE_DIR.parent.parent / ".env"
env_path = root_env if root_env.exists() else BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR / ".env.txt"

if env_path.exists():
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            line = re.sub(r"^\s*", "", line)
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ[key] = val

print(f"📁 Файл окружения: '{env_path}' (Существует? {env_path.exists()})")
print(f"🔑 Проверка памяти: GOOGLE_API_KEY='{os.getenv('GOOGLE_API_KEY')[:10] if os.getenv('GOOGLE_API_KEY') else None}...' | QDRANT='{os.getenv('QDRANT_URL')}'")

qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_docs")

# 2. Ленивая инициализация тяжелых моделей
_model = None
_reranker = None

def get_models():
    """Подгружает модели в память только по требованию и хранит их активными."""
    global _model, _reranker
    if _model is None:
        print("⏳ [Lazy Load] Загрузка BGE-M3 в память...")
        _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    if _reranker is None:
        print("⏳ [Lazy Load] Загрузка реранкера Cross-Encoder...")
        _reranker = CrossEncoder('BAAI/bge-reranker-base')
    return _model, _reranker


def logit_to_percentage(score: float) -> float:
    """Преобразует сырой логит реранкера в проценты от 0 до 100 через сигмоиду."""
    probability = 1 / (1 + math.exp(-score))
    return round(probability * 100, 2)


def rewrite_patient_query(patient_text: str) -> str:
    """
    Универсальный кардиологический оптимизатор поискового запроса.
    Динамически выявляет ведущую патологию (АГ, ИБС/ОКС, Аритмии, ХСН, Липиды)
    и формирует сбалансированную строку терминов: Диагностика + Протоколы лечения.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return patient_text

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    SYSTEM_REWRITE_INSTRUCTION = (
        "Ты — ведущий эксперт-кардиоинформатик и поисковый оптимизатор по Клиническим рекомендациям Минздрава РФ и РКО.\n"
        "Твоя задача — проанализировать анамнез пациента и сформировать единую поисковую строку ключевых терминов, "
        "объединяющую ДИАГНОСТИКУ и ЛЕЧЕНИЕ строго под выявленные у пациента синдромы.\n\n"
        "АЛГОРИТМ ФОРМИРОВАНИЯ ЗАПРОСА:\n"
        "1. Определи ведущие синдромы пациента из областей: Артериальная гипертензия, ИБС/Стенокардия/ОКС, "
        "Нарушения ритма/проводимости (ФП, тахикардии, брадикардии), Сердечная недостаточность (ХСН), Дислипидемия.\n"
        "2. Включи диагностические термины: нозология, стадия/степень, критерии риска (SCORE2, CHA2DS2-VASc), синдромы.\n"
        "3. ОБЯЗАТЕЛЬНО включи термины клинического лечения под выявленные синдромы:\n"
        "   - при гипертензии: целевое АД, алгоритм стартовой терапии, иРААС, БКК, диуретики, немедикаментозные меры;\n"
        "   - при ИБС/ОКС: антиангинальная терапия, двойная антитромбоцитарная терапия (АСК, клопидогрел, тикагрелор), реваскуляризация (ЧКВ);\n"
        "   - при аритмиях/ФП: контроль ритма, контроль ЧСС, антикоагулянтная терапия (ПОАК), катетерная аблация, ЭКС;\n"
        "   - при ХСН: квадротерапия ХСН (валсартан сакубитрил, иНГЛТ-2, спиронолактон), петлевые диуретики, целевой диурез;\n"
        "   - при нарушении липидов: целевой уровень ХС ЛНП, статины высокой интенсивности, эзетимиб.\n\n"
        "КРИТИЧЕСКОЕ ТРЕБОВАНИЕ: Выведи ИСКЛЮЧИТЕЛЬНО строку из 12-20 медицинских ключевых слов через пробел. "
        "Без знаков препинания, без кавычек, без вводных слов."
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_REWRITE_INSTRUCTION}]
        },
        "contents": [{"role": "user", "parts": [{"text": patient_text}]}],
        "generationConfig": {"temperature": 0.1}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"⚠️ Ошибка перефразирования: {e}")
        return patient_text

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"⚠️ Ошибка перефразирования: {e}")
        return patient_text


def hybrid_search(search_query: str, top_k: int = 8, expand_top_neighbors: int = 3):
    embed_model, rerank_model = get_models()

    outputs = embed_model.encode([search_query], return_dense=True, return_sparse=True)
    dense_query = outputs['dense_vecs'][0].tolist()
    sparse_dict = outputs['lexical_weights'][0]

    sparse_query = models.SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()]
    )

    # 1. Извлекаем 50 кандидатов через RRF
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=50),
            models.Prefetch(query=sparse_query, using="sparse", limit=50)
        ],
        query=models.RrfQuery(rrf=models.Rrf()),
        limit=30,
        with_payload=True
    )

    candidate_list = response.points
    if not candidate_list:
        return []

    # 2. Переранжирование через Cross-Encoder
    pairs = [[search_query, hit.payload.get('text', '')] for hit in candidate_list]
    rerank_scores = rerank_model.predict(pairs)
    scored_candidates = sorted(zip(candidate_list, rerank_scores), key=lambda x: x[1], reverse=True)

    # Берем топ-8 лучших чанков
    top_candidates = scored_candidates[:top_k]

    # 3. Подтягивание соседних чанков (Context Window Expansion) для топ-3 лидеров
    expanded_chunks = []
    seen_indices = set()

    for idx, (hit, score) in enumerate(top_candidates):
        payload = hit.payload
        book = payload.get("book")
        c_idx = payload.get("chunk_index")

        # Для лидеров (топ-3) ищем соседей в той же книге через фильтр Qdrant
        if idx < expand_top_neighbors and c_idx is not None and book:
            neighbor_points, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="book", match=models.MatchValue(value=book)),
                        models.FieldCondition(
                            key="chunk_index",
                            range=models.Range(gte=max(1, c_idx - 1), lte=c_idx + 1)
                        )
                    ]
                ),
                limit=3,
                with_payload=True
            )
            # Сортируем соседние чанки по порядку чтения
            neighbor_points.sort(key=lambda p: p.payload.get("chunk_index", 0))

            merged_text = "\n\n".join(
                [p.payload.get("text", "") for p in neighbor_points if p.payload.get("chunk_index") not in seen_indices]
            )
            for p in neighbor_points:
                seen_indices.add(p.payload.get("chunk_index"))

            if merged_text:
                expanded_chunks.append({
                    "text": merged_text,
                    "page": payload.get("page", "Неизвестно"),
                    "score": logit_to_percentage(score)
                })
        else:
            if c_idx not in seen_indices:
                seen_indices.add(c_idx)
                expanded_chunks.append({
                    "text": payload.get("text", ""),
                    "page": payload.get("page", "Неизвестно"),
                    "score": logit_to_percentage(score)
                })

    return expanded_chunks