import json
import os
import time
from dotenv import load_dotenv
from functools import cache
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


class PageMetadata(BaseModel):
    topic: str = Field(
        description="Главная доминирующая тема всей страницы (70-80%+ смыслового объема: название шкалы, матрицы, таблицы или раздела рекомендаций). Категорически запрещено брать тему из изолированных сносок, заголовков 'Пояснения', 'Ключ' или 'Содержание'."
    )
    tags: str = Field(
        description="5-7 ключевых медицинских тегов через запятую, точно отражающих главное содержание всей страницы."
    )


@cache
def get_gemini_client():
    https_proxy = os.getenv('HTTPS_PROXY')
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
    http_proxy = os.getenv('HTTP_PROXY')
    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def refine_medical_chunk(chunk_text):
    client = get_gemini_client()
    model_id = "gemini-3.5-flash-lite"

    sys_instr = (
        "Ты — строгий технический редактор и клинический дата-сайентист.\n"
        "1. ТРАНСКРИБАЦИЯ: Переноси весь фактический текст со всеми цифрами, сносками, ссылками [1] и единицами измерения.\n"
        "2. МЕДИЦИНСКАЯ ТЕРМИНОЛОГИЯ И АББРЕВИАТУРЫ: Сохраняй стандартные аббревиатуры и сокращения в точности, как в оригинале (ЕОК/ЕОАГ, РКО, АГ, САД, ДАД, ЧСС, ХБП, БА, СРТ, ЭКС, SCORE2).\n"
        "   - Если аббревиатура имеет расшифровку в тексте, переноси её корректно.\n"
        "   - Категорически запрещено искажать названия кардиологических организаций, классов рекомендаций (I/II/III, A/B/C) и шкал.\n"
        "3. ЧИСТОТА ФОРМАТИРОВАНИЯ: Делай чистые переносы строк между абзацами, склеивай разорванные внутри предложений слова и предлоги. Не выводи строковые литералы '\\\\n\\\\n'.\n"
        "4. ТАБЛИЦЫ: Переписывай Markdown-таблицы в связный структурированный текст без стрелок ('->', '=>').\n"
        "5. СТРУКТУРА: Сохраняй абзацы и Markdown-заголовки (#, ##)."
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
    )

    max_retries = 20
    attempt = 1

    while attempt <= max_retries:
        try:
            print(f"📝 [Refiner] Очистка текста (Модель: {model_id}, попытка {attempt}/{max_retries})...")
            response = client.models.generate_content(
                model=model_id,
                contents=f"Обработай следующий текст:\n\n{chunk_text}",
                config=config
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ [Refiner] Ошибка API ({model_id}, попытка {attempt}): {e}")
            if attempt == max_retries:
                print(f"❌ [Refiner] Фатальная ошибка после {max_retries} попыток. Возвращаем исходный текст.")
                return chunk_text
            print("🔄 Повтор через 5 сек...")
            time.sleep(5)
            attempt += 1


def extract_page_metadata(full_page_text, previous_topic=None):
    client = get_gemini_client()
    model_id = "gemini-3.7-flash"

    context_hint = ""
    if previous_topic:
        context_hint = f"\nТема предыдущей страницы: '{previous_topic}'. Если текущий текст является продолжением таблицы/темы, сохрани преемственность темы."

    sys_instr = (
        "Ты — аналитик медицинских документов. Определи главную тему и теги для оцифрованной страницы клинического руководства.\n\n"
        "ПРАВИЛА ОПРЕДЕЛЕНИЯ ТЕМЫ (TOPIC):\n"
        "1. ПРИОРИТЕТ ЗАГОЛОВКА: Если в тексте есть Markdown-заголовок (начинается с '### '), возьми его за основу темы. Запрещено выдумывать другие показания/диагнозы, если они не указаны в этом заголовке.\n"
        "2. ПЕРЕЧИСЛЕНИЕ СУБЪЕКТОВ: Если на странице в блоках 'Условия: [Препарат: ...]' описывается несколько препаратов, ОБЯЗАТЕЛЬНО включи их названия в тему.\n"
        "3. ЗАПРЕТ НА МУСОР: Категорически запрещено делать темой слова 'Пояснения', 'Содержание', 'Ключ', 'Введение', 'Сноски'.\n"
        "4. ЗАПРЕТ НА ДОДУМЫВАНИЕ: Не придумывай абстрактные фразы, опирайся строго на видимый текст.\n"
        f"{context_hint}"
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=PageMetadata,
    )

    max_retries = 20
    attempt = 1

    while attempt <= max_retries:
        try:
            print(f"🏷️ [Metadata] Генерация темы и тегов (Модель: {model_id}, попытка {attempt}/{max_retries})...")
            response = client.models.generate_content(
                model=model_id,
                contents=f"Определи тему и теги для следующего текста страницы:\n\n{full_page_text[:4000]}",
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ [Metadata] Ошибка API ({model_id}, попытка {attempt}): {e}")
            if attempt == max_retries:
                print(f"❌ [Metadata] Фатальная ошибка после {max_retries} попыток.")
                return None
            print("🔄 Повтор через 5 сек...")
            time.sleep(5)
            attempt += 1