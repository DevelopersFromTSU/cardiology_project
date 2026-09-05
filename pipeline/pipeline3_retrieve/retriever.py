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
import time

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

qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_docs")

# 2. Ленивая инициализация моделей
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


def rewrite_patient_query(patient_text: str, max_retries: int = 20) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Критическая ошибка: Не задан GOOGLE_API_KEY в окружении.")

    # Используем актуальную рабочую модель (gemini-2.5-flash или gemini-1.5-flash)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    SYSTEM_REWRITE_INSTRUCTION = """
    Ты — ведущий медицинский информатик кардиологических клинических рекомендаций.
    Твоя задача — трансформировать анамнез пациента в сверхточный поисковый вектор из 18–25 ключевых слов и медицинских терминов.

    ПРАВИЛА ПОСТРОЕНИЯ ЗАПРОСА:
    1. ОБЯЗАТЕЛЬНО ВКЛЮЧАЙ ЧИСЛА И ПОКАЗАТЕЛИ:
       - Переноси точные пиковые и пограничные значения: например, "158 120 130", "ЧСС 42", "ЧСС 170", "ФВ 35".
       - Указывай демографический маркер ("пожилые" или конкретный возраст), если это указано в анамнезе.

    2. ПЕРЕВОДИ ЖАЛОБЫ В КЛИНИЧЕСКИЕ СИНДРОМЫ:
       - Повышение АД / кризы -> "артериальная гипертензия 3 степень диастолическая криз органы-мишени"
       - Загрудинная боль / дискомфорт -> "ишемическая болезнь сердца стенокардия напряжения острый коронарный синдром ишемия"
       - Сердцебиение / приступы / перебои -> "пароксизмальная тахикардия фибрилляция предсердий наджелудочковая антиаритмическая"
       - Редкий пульс / обмороки / паузы -> "брадиаритмия синдром слабости синусового узла атриовентрикулярная блокада кардиостимуляция"
       - Одышка / застой / отеки -> "хроническая сердечная недостаточность застой фракция выброса квадротерапия петлевые диуретики"
       - Высокий холестерин / атеросклероз -> "дислипидемия гиперхолестеринемия холестерин ЛНП атеросклероз"

    3. ПОДБИРАЙ КЛАССЫ ПРЕПАРАТОВ И ШКАЛЫ СТРОГО ПОД ВЕДУЩИЙ СИНДРОМ:
       - Для АГ: иРААС антагонисты кальция диуретики SCORE2
       - Для ИБС / ОКС: антиагреганты статины нитраты бета-блокаторы реваскуляризация GRACE
       - Для тахиаритмий / ФП: антиаритмические пульсурежающие антикоагулянты CHA2DS2-VASc
       - Для брадиаритмий: атропин электрокардиостимуляция пейсмейкер
       - Для ХСН: АРНИ иАПФ бета-блокаторы антагонисты альдостерона иНГЛТ-2 диуретики
       - Для дислипидемий: статины эзетимиб ингибиторы PCSK9

    4. СТРОГИЙ ЗАПРЕТ НА АДМИНИСТРАТИВНЫЕ СЛОВА:
       - Категорически ЗАПРЕЩЕНО писать общие слова: "клинические", "рекомендации", "диагностика", "алгоритм", "критерии", "первая линия", "тактика", "РКО", "Минздрав", "пациент". Они засоряют поиск оглавлениями и титульными листами!

    ФОРМАТ ВЫВОДА:
    Строго от 18 до 25 профильных терминов, групп препаратов и чисел в одну строку через пробел.
    """

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_REWRITE_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": patient_text}]}],
        "generationConfig": {"temperature": 0.1}
    }

    attempt = 1
    while attempt <= max_retries:
        try:
            print(f"🔄 [Rewriter] Попытка генерации поискового запроса ({attempt}/{max_retries})...")
            res = requests.post(url, headers=headers, json=payload, timeout=30)

            # Если вернулась ошибка со стороны API (429 лимит, 503 сбой сервера, 404 и т.д.)
            if res.status_code != 200:
                print(f"⚠️ [Rewriter] Сервер API вернул статус {res.status_code}: {res.text}")
                time.sleep(5)
                attempt += 1
                continue

            data = res.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    rewritten = parts[0]["text"].strip()
                    clean_query = " ".join(rewritten.split())
                    print(f"✅ [Rewriter] Поисковый запрос успешно сформирован: {clean_query}")
                    return clean_query

        except requests.exceptions.RequestException as e:
            # Перехват только сетевых таймаутов/разрывов сокета для ухода на повтор, БЕЗ возврата заглушек
            print(f"⚠️ [Rewriter] Ошибка сети при обращении к LLM (попытка {attempt}): {e}")

        time.sleep(5)
        attempt += 1

    # Если после всех попыток ответа нет — прерываем выполнение явной ошибкой, а не мусорной строкой
    raise RuntimeError(f"❌ Фатальный сбой: нейросеть не ответила после {max_retries} попыток.")


def hybrid_search(search_query: str, top_k: int = 10, expand_top_neighbors: int = 3):
    embed_model, rerank_model = get_models()

    outputs = embed_model.encode([search_query], return_dense=True, return_sparse=True)
    dense_query = outputs['dense_vecs'][0].tolist()
    sparse_dict = outputs['lexical_weights'][0]

    sparse_query = models.SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()]
    )

    # 1. Запрос 40 кандидатов в Qdrant
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=80),
            models.Prefetch(query=sparse_query, using="sparse", limit=80)
        ],
        query=models.RrfQuery(rrf=models.Rrf()),
        limit=80,
        with_payload=True
    )

    candidate_list = response.points
    if not candidate_list:
        return []

    # 2. Ранжирование Cross-Encoder
    pairs = [[search_query, hit.payload.get('text', '')] for hit in candidate_list]
    rerank_scores = rerank_model.predict(pairs)
    scored_candidates = sorted(zip(candidate_list, rerank_scores), key=lambda x: x[1], reverse=True)

    # 3. Фильтрация дублей страниц + безопасная склейка контекста
    retrieved_texts = []
    page_counts = {}
    seen_chunk_keys = set()

    for hit, score in scored_candidates:
        payload = hit.payload
        page = payload.get('page', 'Неизвестно')
        book = payload.get('book', 'unknown_book')
        c_idx = payload.get('chunk_index')
        point_id = hit.id

        # Уникальный ключ страницы в рамках конкретной книги
        page_key = f"{book}_{page}"

        # Пропускаем, если с этой страницы уже взяли 2 чанка
        if page_counts.get(page_key, 0) >= 2:
            continue

        # Для топ-3 лидеров подтягиваем соседей (если есть chunk_index)
        if len(retrieved_texts) < expand_top_neighbors and c_idx is not None and book != "unknown_book":
            chunk_key = (book, c_idx)
            if chunk_key in seen_chunk_keys:
                continue

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

            neighbor_points.sort(key=lambda p: p.payload.get("chunk_index", 0))

            texts = []
            pages_set = set()
            for p in neighbor_points:
                n_c_idx = p.payload.get("chunk_index")
                seen_chunk_keys.add((book, n_c_idx))
                texts.append(p.payload.get("text", ""))
                pg = p.payload.get("page")
                if pg:
                    pages_set.add(str(pg))

            merged_text = "\n\n".join(texts).strip()
            pages_str = ", ".join(sorted(pages_set, key=lambda x: int(x) if x.isdigit() else 0)) if pages_set else str(
                page)

            page_counts[page_key] = page_counts.get(page_key, 0) + 1
            retrieved_texts.append({
                "text": merged_text or payload.get("text", ""),
                "page": pages_str,
                "score": logit_to_percentage(score)
            })

        else:
            # Добавление одиночного чанка (для позиций от 4-й и далее или базы без индекса)
            unique_key = (book, c_idx) if c_idx is not None else point_id
            if unique_key not in seen_chunk_keys:
                seen_chunk_keys.add(unique_key)
                page_counts[page_key] = page_counts.get(page_key, 0) + 1
                retrieved_texts.append({
                    "text": payload.get("text", ""),
                    "page": str(page),
                    "score": logit_to_percentage(score)
                })

        # Прерываем цикл строго при достижении необходимого top_k
        if len(retrieved_texts) >= top_k:
            break

    return retrieved_texts
