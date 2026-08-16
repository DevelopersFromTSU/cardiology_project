import json
import os
import time
from dotenv import load_dotenv
from functools import cache
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


class RefinedChunkMetadata(BaseModel):
    topic: str = Field(
        description="Главная доминирующая тема всей страницы (70-80%+ объема текста: например, образ жизни, диета, целевые показатели). Категорически запрещено брать тему из последнего предложения."
    )
    tags: str = Field(
        description="5-7 ключевых тегов, отражающих доминирующую тему страницы."
    )
    refined_text: str = Field(
        description="Полный обработанный текст чанка со 100% сохранением всех цифр, сносок и структуры."
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

# 2. ОБНОВЛЕННАЯ ФУНКЦИЯ: Объединили два промпта в один
def refine_medical_chunk(chunk_text, previous_topic=None, max_retries=3):
    client = get_gemini_client()
    model_id = "gemini-3.5-flash-lite"

    # Добавляем контекст из старого генератора метаданных
    context_hint = ""
    if previous_topic:
        context_hint = f"\nДля справки, тема предыдущей страницы была: '{previous_topic}'. Если текущий текст выглядит как продолжение, учти это при формулировании темы."

    sys_instr = (
        "Ты — строгий технический редактор и клинический дата-сайентист.\n\n"
        "1. ТРАНСКРИБАЦИЯ: Переноси абсолютно весь текст на 100% со всеми цифрами, сносками и единицами измерения.\n"
        "2. ТАБЛИЦЫ: Переписывай Markdown-таблицы в связный текст без математических стрелок ('->', '=>').\n"
        "3. СТРУКТУРА: Сохраняй абзацы (\\n\\n) и Markdown-заголовки (#, ##).\n"
        "4. АББРЕВИАТУРЫ: Не изменяй и не удаляй медицинские сокращения.\n\n"
        "5. ПРАВИЛО ВЫБОРА ТЕМЫ (TOPIC) И ТЕГОВ (TAGS) — КРИТИЧНО:\n"
        "   - Если в тексте встречается заголовок или выделенная фраза раздела (например, 'Советы пациенту и его семье', 'Диагностика...', 'Лечение...'), ты ОБЯЗАН взять тему из этого заголовка.\n"
        "   - Если явного заголовка нет, формулируй тему по ПЕРВОЙ ПОЛОВИНЕ текста (первые 2-3 абзаца).\n"
        "   - СТРОЖАЙШЕ ЗАПРЕЩЕНО брать тему из последнего предложения или изолированных сносок внизу страницы.\n"
        "   - В 'tags' перечисли термины, описывающие основное полотно текста (образ жизни, диета, соль, физическая активность, целевые уровни)."
        f"{context_hint}"
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.1,  # Немного подняли для лучшей генерации тегов
        response_mime_type="application/json",
        response_schema=RefinedChunkMetadata,
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

# ФУНКЦИЮ generate_page_metadata УДАЛЯЕМ ПОЛНОСТЬЮ!