import re
import requests
import json
import os
from dotenv import load_dotenv
from google import genai
from utils.abbreviations import force_expand_abbreviations
import time

load_dotenv()

def get_gemini_client():
    os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
    os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def refine_medical_chunk(chunk_text, max_retries=3):
    client = get_gemini_client()
    model_id = "gemini-2.5-flash"  # Рекомендую 2.0, так как на него у тебя настроены квоты

    sys_instr = (
        "Ты — строгий технический редактор и медицинский аналитик. "
        "Твоя задача:\n"
        "1. ИСПРАВЛЕНИЕ: Устрани опечатки и грамматические ошибки.\n"
        "2. ПРЕОБРАЗОВАНИЕ ТАБЛИЦ: Если в тексте есть Markdown-таблицы (|---|), ПЕРЕПИШИ их в виде логических цепочек "
        "со стрелками `->`."
        "Каждая строка должна быть самодостаточной: сочетай заголовок строки, заголовок столбца и значение в одно "
        "предложение.\n"
        "3. СОХРАНЕНИЕ СТРУКТУРЫ: Строго сохраняй вложенность списков, стрелки и ВСЕ Markdown-заголовки (#, ##, "
        "###). Не удаляй и не изменяй уровень заголовков.\n"
        "4. ФОРМАТ: Верни результат СТРОГО в формате JSON."
        "5. ФОРМАТИРОВАНИЕ ТЕКСТА: Используй `\\n\\n` только для разделения абзацев и пунктов списка. Внутри одного "
        "предложения или одного логического пункта списка НЕ ДОЛЖНО БЫТЬ никаких переносов строк (`\\n`). Текст "
        "пункта должен идти сплошной строкой."
    )

    json_prompt = """
    {
        "refined_text": "исправленный текст"
    }
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                # ЗАМЕНИ expanded_text на chunk_text
                contents=f"Обработай следующий текст и верни его в формате {json_prompt}:\n\n{chunk_text}",
                # ИСПРАВЛЕНИЕ 2: Исправили отступы
                config={
                    "system_instruction": sys_instr,
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            )

            data = json.loads(response.text)

            return data

        except Exception as e:
            # ИСПРАВЛЕНИЕ 3: Починили логику повторов
            print(f"⚠️ Ошибка Gemini (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(15)  # Ждем перед следующей попыткой

    # Если цикл закончился, а return data не сработал
    print("❌ Не удалось обработать текст после всех попыток.")
    return None
