import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pipeline.utils.abbreviations import MEDICAL_DICT

load_dotenv()

DICT_PROMPT_STRING = "\n".join([f"{k} -> {v}" for k, v in MEDICAL_DICT.items()])


class ExtractedFact(BaseModel):
    context: str = Field(
        description="Полный набор условий, осей или заголовков (например: 'Препарат: Агонисты имидазолиновых рецепторов, Категория: Противопоказания')")
    value: str = Field(description="Конкретное значение, цифра, результат или симптом")


# [НОВОЕ]: Строгая модель для ячейки строки вместо динамического dict, чтобы обойти ограничение additionalProperties
class RowStateCell(BaseModel):
    column_name: str = Field(
        description="Название колонки или заголовка (например 'Препарат', 'Показания', 'Дозировка')")
    cell_value: str = Field(description="Точный текст ячейки в этой колонке в самом низу изображения")


class ImageExtraction(BaseModel):
    analysis_status: str = Field(description="Статус: 'success' если данные успешно извлечены, иначе 'failed'")
    source_type: str = Field(description="Тип контента: 'таблица', 'алгоритм', 'график' или 'легенда'")
    global_context: str = Field(description="Общее название или суть изображения")

    # [ИСПРАВЛЕНО]: Используем list[RowStateCell] вместо dict[str, str], так как Gemini API не поддерживает additionalProperties
    last_row_state: list[RowStateCell] = Field(
        default=[],
        description="Снимок всей крайней нижней строки таблицы в виде списка колонок и их значений в самом низу страницы."
    )
    facts: list[ExtractedFact] = Field(description="Массив всех извлеченных атомарных данных")
    footnotes: list[str] = Field(
        default=[],
        description="Дословный список всех сносок, примечаний и пояснений под таблицей, обозначенных звездочками (*), буквами (a, b, c) или цифрами (1, 2)"
    )


def describe_image(pil_img, full_page_img=None, previous_table_title=None, previous_row_state=None):
    if pil_img is None:
        return "", None, {}

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    continuation_prompt = ""
    if previous_table_title:
        continuation_prompt += f"\nВНИМАНИЕ: Это изображение может быть продолжением таблицы '{previous_table_title}'."

    # Передача снимка многоколоночной таблицы в JSON-формате
    if previous_row_state and isinstance(previous_row_state, dict) and len(previous_row_state) > 0:
        state_json_str = json.dumps(previous_row_state, ensure_ascii=False)
        continuation_prompt += (
            f"\nКРИТИЧЕСКОЕ ПРАВИЛО МНОГОКОЛОНОЧНОГО ПЕРЕНОСА: Верхние ячейки таблицы могут быть продолжением строки с прошлой страницы. "
            f"Вот полный снимок (словарь колонок) окончания прошлой страницы: {state_json_str}. "
            f"Если ячейка слева пуста, подставь субъект из этого словаря. Если ячейка в любой из колонок начинается со строчной буквы или обрывка слова, "
            f"найди соответствующую колонку в словаре и бесшовно объедини окончание с прошлой страницы с началом на этой странице!"
        )

    sys_instr = (
        "Ты — эксперт-аналитик по оцифровке медицинских данных для векторных баз (RAG). "
        "Твоя задача — извлечь все факты из изображения и разбить их на пары 'Условие -> Значение'.\n\n"
        "ПРАВИЛА:\n"
        "1. ОРФОГРАФИЯ И МОРФОЛОГИЯ: Пиши с безупречной грамотностью, автоматически исправляя опечатки и склейки пробелов оригинального скана (например, 'всочетании' -> 'в сочетании'). Если на изображении есть медицинские аббревиатуры, "
        "ОБЯЗАТЕЛЬНО расшифровывай их согласно предоставленному ниже словарю, СТРОГО согласуя падеж, род и число с контекстом.\n"
        "2. АТОМАРНОСТЬ И ВЕРТИКАЛЬНО ОБЪЕДИНЕННЫЕ ЯЧЕЙКИ (ROWSPAN): Каждый факт независим. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО сокращать или группировать матрицы! "
        "КРИТИЧЕСКИ ВАЖНО: Внимательно анализируй визуальную структуру сетки. Если в любом столбце таблицы ячейка объединена по вертикали на несколько строк вниз (между строками отсутствует горизонтальная разделительная линия), "
        "ты ОБЯЗАН наследовать значение этой ячейки сверху вниз на весь диапазон объединения. Создавай отдельный самостоятельный факт для КАЖДОЙ логической строки, входящей в этот блок, дублируя общее значение. "
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО пропускать строки, оставлять их значения пустыми или писать '[Пусто]', если визуально они находятся внутри вертикально объединенной зоны!\n"
        "3. НУЛЕВАЯ ПОТЕРЯ ДАННЫХ: Извлекай текст таблиц дословно по смыслу и словам, с правильной орфографией. Не выдумывай несуществующие факты и цифры. Неразборчиво = пиши '[Неразборчиво]'.\n"
        "4. КРИТИЧЕСКОЕ ПРАВИЛО СНОСОК И ПРИМЕЧАНИЙ (FOOTNOTES): Внимательно сканируй нижнюю часть изображения под сеткой таблицы! Часто там находятся важнейшие клинические сноски и исключения.\n"
        "   - ОБЯЗАТЕЛЬНО ищи строки, начинающиеся со значка звездочки (`*`, `**`), надстрочных букв (например, `a`, `б`, `c`) или надстрочных цифр (`1`, `2`), которые ссылаются на аналогичные маркеры внутри ячеек таблицы.\n"
        "   - Извлеки текст каждой сноски абсолютно ДОСЛОВНО (включая ссылки на источники в квадратных скобках, например `[161, 227, 128]`) и помести их в массив `footnotes` в формате: `* Текст сноски` или `a Текст сноски`.\n"
        "   - Если изображение (кроп) обрезано впритык к низу таблицы, обязательно сверься с изображением полной страницы (первым переданным изображением), чтобы убедиться, что ни одна сноска под таблицей не была пропущена!\n"
        "   - Если сносок графически нет вообще — оставь массив `footnotes` строго пустым.\n"
        f"{continuation_prompt}\n\n"
        f"--- СЛОВАРЬ РАСШИФРОВКИ АББРЕВИАТУР ---\n{DICT_PROMPT_STRING}"
    )

    user_content = []
    if full_page_img is not None:
        user_content.extend([full_page_img, pil_img,
                             "Оцифруй второе изображение (кроп) в массив строгих фактов с учетом переноса таблиц."])
    else:
        user_content.extend([pil_img, "Оцифруй данное изображение в массив фактов с учетом переноса таблиц."])

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        max_output_tokens=32768,
        response_mime_type="application/json",
        response_schema=ImageExtraction
    )

    try:
        print("⏳ Оцифровка таблицы с контролем многоколоночных разрывов...")
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=user_content,
            config=config
        )

        data = json.loads(response.text)

        if data.get("analysis_status") == "success" and (data.get("facts") or data.get("footnotes")):
            global_ctx = data.get("global_context", "").strip()
            source_type = data.get("source_type", "")

            # [НОВОЕ]: Преобразуем список объектов RowStateCell обратно в удобный словарь Python
            raw_row_list = data.get("last_row_state", [])
            row_state = {}
            if isinstance(raw_row_list, list):
                for cell in raw_row_list:
                    if isinstance(cell, dict) and "column_name" in cell and "cell_value" in cell:
                        row_state[cell["column_name"]] = cell["cell_value"]

            # Проверка завершенности предложения
            non_empty_vals = [str(val).strip() for val in row_state.values() if str(val).strip()]
            if non_empty_vals and all(val.endswith((".", "!", "?", ";")) for val in non_empty_vals):
                row_state = {}

            current_table_title = None

            if source_type == "таблица":
                is_continuation = "продолжение" in global_ctx.lower() or len(global_ctx) < 10
                if is_continuation and previous_table_title:
                    global_ctx = f"{previous_table_title} (Продолжение)"
                    current_table_title = previous_table_title
                else:
                    current_table_title = global_ctx

            text_blocks = [f"--- Контекст изображения: {global_ctx} ({source_type}) ---"]

            for fact in data.get("facts", []):
                text_blocks.append(f"Условия: [{fact['context']}] => Значение: {fact['value']}")

            if data.get("footnotes"):
                text_blocks.append("\n--- Сноски и примечания к таблице ---")
                for note in data["footnotes"]:
                    text_blocks.append(f"• {note}")

            final_text = "\n".join(text_blocks)
            return final_text, current_table_title, row_state
        else:
            return "", None, {}

    except Exception as e:
        print(f"❌ Ошибка при обработке картинки: {e}")
        return "", None, {}