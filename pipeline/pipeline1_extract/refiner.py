import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functools import cache
from google import genai

load_dotenv()


@cache
def get_gemini_client():
    os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
    os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def refine_medical_chunk(chunk_text, max_retries=3):
    client = get_gemini_client()
    model_id = "gemini-2.5-flash"

    # [ИСПРАВЛЕНО]: Пункт 1 теперь требует от модели самостоятельно находить сокращения из словаря
    # и сразу встраивать их расшифровки в правильном падеже.
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
        "4. ФОРМАТ: Верни результат СТРОГО в формате JSON. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО оборачивать ответ в маркдаун-блоки "
        "```json ... ```. Начни ответ сразу с { и закончи }.\n"
        "5. ФОРМАТИРОВАНИЕ ТЕКСТА: Используй \\n\\n только для разделения абзацев. Внутри одного "
        "предложения или одного логического пункта списка НЕ ДОЛЖНО БЫТЬ никаких переносов строк (\\n).\n"
        "6. ПРАВИЛО ПОЛНОТЫ: Работай в режиме дословного транскрибатора. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО сокращать или "
        "пропускать слова. Переноси абсолютно весь текст на 100% до единой буквы.\n\n"
    )
    json_prompt = """
    {
        "refined_text": "исправленный текст"
    }
    """

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.1,
        response_mime_type="application/json"
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=f"Обработай следующий текст и верни его в формате {json_prompt}:\n\n{chunk_text}",
                config=config
            )

            text_content = response.text

            cleaned_text = text_content.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]

            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            cleaned_text = cleaned_text.strip()

            data = json.loads(cleaned_text)
            return data

        except Exception as e:
            print(f"⚠️ Ошибка Gemini (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(5)

    print("❌ Не удалось обработать текст после всех попыток.")
    return None


def generate_page_metadata(page_text, previous_topic=None, max_retries=3):
    """
    Анализирует собранный текст страницы и извлекает тему и теги.
    """
    client = get_gemini_client()
    # ИСПРАВЛЕНИЕ 1: Используем существующую версию модели
    model_id = "gemini-2.5-flash"

    context_hint = ""
    if previous_topic:
        context_hint = f"\nДля справки, тема предыдущей страницы была: '{previous_topic}'. Если текущий текст выглядит как продолжение (например, обрывок таблицы), учти это при формулировании новой темы и укажи, что это продолжение."

    json_prompt = """
    {
        "topic": "Краткая тема страницы...",
        "tags": "тег1, тег2, тег3, тег4, тег5"
    }
    """

    # ИСПРАВЛЕНИЕ 2: Внедряем json_prompt в системную инструкцию
    sys_instr = (
        "Ты — клинический дата-сайентист. Твоя задача — проанализировать извлеченный текст страницы медицинской книги "
        "и выделить ее главный смысловой контекст для поисковой системы.\n"
        "1. Сформулируй главную тему страницы предельно кратко, емко и структурировано (как заголовок статьи или справочника). "
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать вводные слова, полные предложения и конструкции вроде 'Эта страница представляет собой...', 'Здесь описывается...' или 'Таблица показывает...'. "
        "Пиши только саму суть (например: 'Методы обследования и пороговые значения для выявления поражения органов-мишеней (ПОМ)').\n"
        "2. Выдели 5-7 ключевых медицинских тегов (симптомы, диагнозы, показатели, препараты), которые встречаются в тексте.\n"
        "3. Верни результат СТРОГО в формате JSON без использования markdown-разметки (```json ... ```).\n"
        f"{context_hint}\n"
        f"ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА:\n{json_prompt}"
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.1,
        response_mime_type="application/json"
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=f"Текст страницы:\n\n{page_text}",
                config=config
            )

            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            data = json.loads(cleaned_text.strip())
            return data

        except Exception as e:
            print(f"⚠️ Ошибка генерации метаданных (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)

    return {"topic": "Медицинские данные", "tags": "медицина, справочник"}