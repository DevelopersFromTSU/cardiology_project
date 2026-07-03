import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
# [НОВОЕ]: Импортируем наш словарь аббревиатур напрямую, как в vision.py
from pipeline.utils.abbreviations import MEDICAL_DICT

load_dotenv()

# [НОВОЕ]: Формируем строковое представление словаря для промпта
DICT_PROMPT_STRING = "\n".join([f"{k} -> {v}" for k, v in MEDICAL_DICT.items()])


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
        "Ты — строгий технический редактор и медицинский аналитик. "
        "Твоя задача:\n"
        "1. МОРФОЛОГИЯ И РАСШИФРОВКА АББРЕВИАТУР: Исправь любые опечатки и неверные буквы в словах. "
        "Если в тексте встречаются медицинские аббревиатуры из предоставленного ниже словаря, ОБЯЗАТЕЛЬНО расшифровывай их. "
        "При расшифровке СТРОГО согласуй падежные окончания, число и род с предлогами и контекстом предложения "
        "(например, сокращение в фразе 'на фоне АГ' расшифруй и просклоняй как 'на фоне артериальной гипертензии (АГ)').\n"

        "2. ПРЕОБРАЗОВАНИЕ ТАБЛИЦ: Если в тексте есть Markdown-таблицы (|---|), ПЕРЕПИШИ их в виде развернутых, "
        "связных предложений естественным языком БЕЗ ИСПОЛЬЗОВАНИЯ СИМВОЛОВ СТРЕЛОК ('->', '=>'). "
        "Каждую ячейку преобразуй по схеме: 'В таблице [Название] для строки [Название строки] в столбце [Название "
        "столбца] указано значение: [Значение]'. Каждое предложение должно быть самодостаточным.\n"

        "3. СОХРАНЕНИЕ СТРУКТУРЫ: Строго сохраняй вложенность списков и ВСЕ Markdown-заголовки (#, ##, "
        "###). Не удаляй и не изменяй уровень заголовков.\n"
        "4. ФОРМАТ: Верни результат СТРОГО в формате JSON. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО оборачивать ответ в маркдаун-блоки "
        "```json ... ```. Начни ответ сразу с { и закончи }.\n"
        "5. ФОРМАТИРОВАНИЕ ТЕКСТА: Используй `\\n\\n` только для разделения абзацев и пунктов списка. Внутри одного "
        "предложения или одного логического пункта списка НЕ ДОЛЖНО БЫТЬ никаких переносов строк (`\\n`). Текст "
        "пункта должен идти сплошной строкой.\n"
        "6. ПРАВИЛО ПОЛНОТЫ: Работайте в режиме дословного транскрибатора. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО сокращать, "
        "резюмировать, объединять или пропускать слова, символы и сноски. Переноси абсолютно весь текст на 100% до "
        "единой буквы, даже если предложение обрывается на полуслове.\n\n"
        f"--- СЛОВАРЬ РАСШИФРОВКИ АББРЕВИАТУР ---\n{DICT_PROMPT_STRING}"
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