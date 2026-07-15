import json
import os
import time
from dotenv import load_dotenv
from functools import cache
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

class RefinedText(BaseModel):
    refined_text: str

class PageMetadata(BaseModel):
    topic: str
    tags: str


@cache
def get_gemini_client():
    https_proxy = os.getenv('HTTPS_PROXY')
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy

    http_proxy = os.getenv('HTTP_PROXY')
    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy

    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def refine_medical_chunk(chunk_text, max_retries=3):
    client = get_gemini_client()
    model_id = "gemini-2.5-flash"

    # [ИСПРАВЛЕНО]: Убрали ручные инструкции по формату JSON
    sys_instr = (
        "Ты — строгий технический редактор и медицинский аналитик. Твоя задача:\n"
        "1. ОРФОГРАФИЯ: Исправь опечатки OCR.\n"
        "2. ПРЕОБРАЗОВАНИЕ ТАБЛИЦ И СПИСКОВ (СТРАХОВОЧНЫЙ ШАГ): Если парсер захватил таблицу в виде Markdown-текста (|---|), "
        "перепиши её в связные предложения естественным языком. "
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать символы математических стрелок ('->', '=>'). "
        "Если перед тобой медицинский перечень или список симптомов, преобразуй каждый пункт в текстовый формат: "
        "'Условие: [...]. Значение: [...]', чтобы текст не был сплошным неструктурированным полотном.\n"
        "3. СОХРАНЕНИЕ СТРУКТУРЫ: Строго сохраняй вложенность списков и ВСЕ Markdown-заголовки (#, ##, ###). "
        "Не изменяй уровень заголовков.\n"
        "4. ФОРМАТИРОВАНИЕ ТЕКСТА: Используй \\n\\n только для разделения абзацев. Внутри одного "
        "предложения или одного логического пункта списка НЕ ДОЛЖНО БЫТЬ никаких переносов строк (\\n).\n"
        "5. ПРАВИЛО ПОЛНОТЫ: Работай в режиме дословного транскрибатора. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО сокращать или "
        "пропускать слова. Переноси абсолютно весь текст на 100% до единой буквы.\n\n"
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=RefinedText,  # Принудительная структура
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=f"Обработай следующий текст:\n\n{chunk_text}",
                config=config
            )
            return json.loads(response.text)

        except Exception as e:
            print(f"⚠️ Ошибка Gemini (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(5)

    print("❌ Не удалось обработать текст после всех попыток.")
    return None


def generate_page_metadata(page_text, previous_topic=None, max_retries=3):
    client = get_gemini_client()
    model_id = "gemini-2.5-flash"

    context_hint = ""
    if previous_topic:
        context_hint = f"\nДля справки, тема предыдущей страницы была: '{previous_topic}'. Если текущий текст выглядит как продолжение (например, обрывок таблицы), учти это при формулировании новой темы и укажи, что это продолжение."

    sys_instr = (
        "Ты — клинический дата-сайентист. Твоя задача — проанализировать извлеченный текст страницы медицинской книги "
        "и выделить ее главный смысловой контекст для поисковой системы.\n"
        "1. Сформулируй главную тему страницы предельно кратко, емко и структурировано (как заголовок статьи или справочника). "
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать вводные слова, полные предложения. Пиши только саму суть.\n"
        "2. Выдели 5-7 ключевых медицинских тегов (симптомы, диагнозы, показатели, препараты), которые встречаются в тексте.\n"
        f"{context_hint}\n"
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=PageMetadata,  # Строгое ограничение
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=f"Текст страницы:\n\n{page_text}",
                config=config
            )
            return json.loads(response.text)

        except Exception as e:
            print(f"⚠️ Ошибка генерации метаданных (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)

    return {"topic": "Медицинские данные", "tags": "медицина, справочник"}