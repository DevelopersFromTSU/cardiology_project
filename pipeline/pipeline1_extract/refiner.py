import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def get_gemini_client():
    # Прокси оставляем, если они нужны для сети
    os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
    os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def refine_medical_chunk(chunk_text, max_retries=3):
    client = get_gemini_client()
    model_id = "gemini-flash-latest"  # Можно использовать gemini-2.5-flash для скорости

    sys_instr = (
        "Ты — строгий технический редактор и медицинский аналитик. "
        "Твоя задача:\n"
        "1. ИСПРАВЛЕНИЕ: Устрани опечатки и грамматические ошибки.\n"

        "2. ПРЕОБРАЗОВАНИЕ ТАБЛИЦ: Если в тексте есть Markdown-таблицы (|---|), ПЕРЕПИШИ их в виде развернутых, "
        "связных предложений естественным языком БЕЗ ИСПОЛЬЗОВАНИЯ СИМВОЛОВ СТРЕЛОК ('->', '=>')."
        "Каждую ячейку преобразуй по схеме: 'В таблице [Название] для строки [Название строки] в столбце [Название "
        "столбца] указано значение: [Значение]'."
        "Каждое предложение должно быть самодостаточным.\n"

        "3. СОХРАНЕНИЕ СТРУКТУРЫ: Строго сохраняй вложенность списков и ВСЕ Markdown-заголовки (#, ##, "
        "###). Не удаляй и не изменяй уровень заголовков.\n"
        "4. ФОРМАТ: Верни результат СТРОГО в формате JSON. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО оборачивать ответ в маркдаун-блоки "
        "```json ... ```. Начни ответ сразу с { и закончи }.\n"
        "5. ФОРМАТИРОВАНИЕ ТЕКСТА: Используй `\\n\\n` только для разделения абзацев и пунктов списка. Внутри одного "
        "предложения или одного логического пункт списка НЕ ДОЛЖНО БЫТЬ никаких переносов строк (`\\n`). Текст "
        "пункта должен идти сплошной строкой.\n"
        "6. ПОЛНОТА ТЕКСТА: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО удалять, пропускать или обрезать любые слова, даже если предложение"
        "в самом конце обрывается на полуслове. Переноси абсолютно всё до единого символа."
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

            # Твой блок надежной очистки от маркдауна (отлично работает, оставляем)
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