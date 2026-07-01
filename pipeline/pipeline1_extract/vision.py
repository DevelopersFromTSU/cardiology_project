import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


# =====================================================================
# 1. СТРОГИЕ УНИВЕРСАЛЬНЫЕ СХЕМЫ ДАННЫХ ДЛЯ ЛЮБОГО ИЗОБРАЖЕНИЯ
# =====================================================================

class ExtractedFact(BaseModel):
    context: str = Field(
        description="Полный набор условий, осей или заголовков (например: 'Пол: Ж, Возраст: 50, Давление: 140' или 'Шаг алгоритма: Остановка сердца')")
    value: str = Field(description="Конкретное значение, цифра, результат или следующее действие для данного контекста")


class ImageExtraction(BaseModel):
    analysis_status: str = Field(description="Статус: 'success' если данные успешно извлечены, иначе 'failed'")
    source_type: str = Field(description="Тип контента: 'таблица', 'алгоритм', 'график' или 'легенда'")
    global_context: str = Field(description="Общее название или суть изображения")
    facts: list[ExtractedFact] = Field(description="Массив всех извлеченных атомарных данных")


# =====================================================================

def describe_image(pil_img, full_page_img=None):
    if pil_img is None:
        return ""

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Промпт теперь короткий, обобщенный и направлен только на логику извлечения
    sys_instr = (
        "Ты — эксперт-аналитик по оцифровке медицинских данных для векторных баз (RAG). "
        "Твоя задача — извлечь все факты из изображения и разбить их на пары 'Условие -> Значение'.\n\n"
        "ПРАВИЛА:\n"
        "1. УНИВЕРСАЛЬНОСТЬ: Адаптируйся под любой контент. Если это таблица — скрещивай строки и столбцы. "
        "Если алгоритм — описывай логические переходы. Если график — точки и оси.\n"
        "2. АТОМАРНОСТЬ: Каждый факт независим. В поле 'context' дублируй ВСЕ параметры, которые ведут к значению. "
        "В поле 'value' пиши только итоговую цифру или действие.\n"
        "3. ПОЛНОТА: Не пропускай ячейки, не группируй данные. 100 ячеек = 100 объектов фактов.\n"
        "4. БЕЗ ФАНТАЗИЙ: Переноси данные один в один. Неразборчиво = пиши '[Неразборчиво]'."
        "5. ЗАПРЕТ НА ДУБЛИРОВАНИЕ: Анализируй каждый визуальный блок (таблицу, легенду, график) ровно один раз. "
        "Категорически запрещено извлекать одни и те же правила или цифры повторно, используя другие формулировки. "
        "Одно уникальное правило на картинке = один объект в массиве."
    )

    user_content = []

    if full_page_img is not None:
        text_prompt = (
            "Используй первое изображение (всю страницу) для понимания глобального контекста и легенды. "
            "Оцифруй второе изображение (кроп) в массив строгих фактов."
        )
        user_content.extend([full_page_img, pil_img, text_prompt])
    else:
        text_prompt = "Оцифруй данное изображение в массив фактов."
        user_content.extend([pil_img, text_prompt])

    # Привязываем Pydantic-схему к конфигу
    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,  # Температура 0 для жесткой детерминированности
        response_mime_type="application/json",
        response_schema=ImageExtraction
    )

    try:
        print("⏳ Оцифровка через Structured Outputs...")
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=user_content,
            config=config
        )

        data = json.loads(response.text)

        if data.get("analysis_status") == "success" and data.get("facts"):
            # Формируем сверхплотный текст, идеальный для модели BGE-M3
            text_blocks = [f"--- Контекст изображения: {data.get('global_context')} ({data.get('source_type')}) ---"]

            for fact in data.get("facts", []):
                # Формат, который отлично бьется на чанки и сохраняет семантику
                fact_str = f"Условия: [{fact['context']}] => Значение: {fact['value']}"
                text_blocks.append(fact_str)

            final_text = "\n".join(text_blocks)
            return final_text
        else:
            return ""

    except Exception as e:
        print(f"❌ Ошибка при обработке картинки: {e}")
        return ""