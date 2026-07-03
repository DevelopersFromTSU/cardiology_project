import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
# [ИСПРАВЛЕНО]: Импортируем именно словарь MEDICAL_DICT напрямую из модуля
from pipeline.utils.abbreviations import MEDICAL_DICT

load_dotenv()

# [ИСПРАВЛЕНО]: Формируем чистый текст словаря без ошибок импорта и без регулярных выражений
DICT_PROMPT_STRING = "\n".join([f"{k} -> {v}" for k, v in MEDICAL_DICT.items()])


class ExtractedFact(BaseModel):
    context: str = Field(
        description="Полный набор условий, осей или заголовков (например: 'Пол: Ж, Возраст: 50, Давление: 140')")
    value: str = Field(description="Конкретное значение, цифра, результат или следующее действие для данного контекста")


class ImageExtraction(BaseModel):
    analysis_status: str = Field(description="Статус: 'success' если данные успешно извлечены, иначе 'failed'")
    source_type: str = Field(description="Тип контента: 'таблица', 'алгоритм', 'график' или 'легенда'")
    global_context: str = Field(description="Общее название или суть изображения")
    facts: list[ExtractedFact] = Field(description="Массив всех извлеченных атомарных данных")
    footnotes: list[str] = Field(default=[], description="Дословный список всех сносок и примечаний под таблицей")


def describe_image(pil_img, full_page_img=None, previous_table_title=None):
    if pil_img is None:
        return "", None

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    sys_instr = (
        "Ты — эксперт-аналитик по оцифровке медицинских данных для векторных баз (RAG). "
        "Твоя задача — извлечь все факты из изображения и разбить их на пары 'Условие -> Значение'.\n\n"
        "ПРАВИЛА:\n"
        "1. УНИВЕРСАЛЬНОСТЬ И МОРФОЛОГИЯ: Адаптируйся под любой контент. Если на изображении есть медицинские аббревиатуры, "
        "ОБЯЗАТЕЛЬНО расшифровывай их согласно предоставленному ниже словарю. При расшифровке СТРОГО согласуй падеж, род и число "
        "с контекстом предложения (например, если на картинке 'при низкой ФВ', пиши 'при низкой фракцией выброса (ФВ)').\n"
        "2. АТОМАРНОСТЬ: Каждый факт независим. В поле 'context' дублируй ВСЕ параметры, которые ведут к значению. "
        "В поле 'value' пиши только итоговую цифру или действие.\n"
        "3. НУЛЕВАЯ ПОТЕРЯ ДАННЫХ И СНОСОК: Извлекай текст verbatim (дословно). Все примечания, мелкий шрифт "
        "и текст под звездочками (*, **) ОБЯЗАТЕЛЬНО помещай в массив 'footnotes'.\n"
        "4. БЕЗ ФАНТАЗИЙ: Переноси данные один в один. Неразборчиво = пиши '[Неразборчиво]'.\n\n"
        f"--- СЛОВАРЬ РАСШИФРОВКИ АББРЕВИАТУР ---\n{DICT_PROMPT_STRING}"
    )

    user_content = []
    if full_page_img is not None:
        text_prompt = "Используй первое изображение (всю страницу) для понимания контекста. Оцифруй второе изображение (кроп) в массив строгих фактов и сносок с расшифровкой аббревиатур."
        user_content.extend([full_page_img, pil_img, text_prompt])
    else:
        text_prompt = "Оцифруй данное изображение в массив фактов и сносок с расшифровкой аббревиатур."
        user_content.extend([pil_img, text_prompt])

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=ImageExtraction
    )

    try:
        print("⏳ Оцифровка через Structured Outputs...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=config
        )

        data = json.loads(response.text)

        if data.get("analysis_status") == "success" and (data.get("facts") or data.get("footnotes")):
            global_ctx = data.get("global_context", "").strip()
            source_type = data.get("source_type", "")
            current_table_title = None

            if source_type == "таблица":
                is_continuation = "продолжение" in global_ctx.lower() or len(global_ctx) < 10
                if is_continuation and previous_table_title:
                    global_ctx = f"{previous_table_title} (Продолжение с предыдущей страницы)"
                    current_table_title = previous_table_title
                else:
                    current_table_title = global_ctx

            text_blocks = [f"--- Контекст изображения: {global_ctx} ({source_type}) ---"]

            for fact in data.get("facts", []):
                fact_str = f"Условия: [{fact['context']}] => Значение: {fact['value']}"
                text_blocks.append(fact_str)

            if data.get("footnotes"):
                text_blocks.append("\n--- Сноски и примечания к таблице ---")
                for note in data["footnotes"]:
                    text_blocks.append(f"• {note}")

            final_text = "\n".join(text_blocks)
            return final_text, current_table_title
        else:
            return "", None

    except Exception as e:
        print(f"❌ Ошибка при обработке картинки: {e}")
        return "", None