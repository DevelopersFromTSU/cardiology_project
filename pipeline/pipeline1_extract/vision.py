import os
import json
import time
from enum import Enum
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

load_dotenv()


class ImageCategory(str, Enum):
    FLOWCHART = "flowchart"
    MATRIX = "matrix"
    TEXT_TABLE = "text_table"


class ImageRouter(BaseModel):
    category: ImageCategory = Field(
        description="Категория медицинского изображения: 'flowchart', 'matrix', 'text_table'."
    )


PROMPT_MATRIX = (
    "Перед тобой плотная многомерная матрица (шкала рисков SCORE2, стратификация, сетка дозировок).\n"
    "ТВОЯ ЦЕЛЬ: Подготовить изолированные чанки данных для RAG-системы. Точность координат критична.\n"
    "1. ТОЧНЫЕ КООРДИНАТЫ: В 'context' собери путь по оси Y. В 'value' ОБЯЗАТЕЛЬНО привяжи цифры к их колонкам (ось X). Ищи вертикальные заголовки осей.\n"
    "2. ГОРИЗОНТАЛЬНАЯ ГРУППИРОВКА: Собери весь ряд значений по горизонтали в одно поле 'value' с соблюдением осей.\n"
    "3. ИЗОЛЯЦИЯ: Каждая логическая строка матрицы — это один независимый объект JSON.\n"
    "4. ПУСТЫЕ ЗОНЫ: Игнорируй пустые ячейки.\n"
    "5. ЕДИНИЦЫ ИЗМЕРЕНИЯ: Обязательно прикрепляй единицы измерения (%, мг, ммоль/л) к 'голым' цифрам."
)

PROMPT_FLOWCHART = (
    "Перед тобой медицинское визуальное представление (алгоритм, схема лечения, цикл или анатомическая схема).\n"
    "ТВОЯ ЦЕЛЬ: Оцифровать данные для RAG-системы 'Ключ -> Значение'.\n"
    "1. АДАПТИВНОЕ ИЗВЛЕЧЕНИЕ: В 'context' пиши [Родитель -> Условие перехода], в 'value' — [Целевой блок/Действие].\n"
    "2. ОПТИМИЗАЦИЯ: Объединяй общие условия. Не дроби списки препаратов.\n"
    "3. АББРЕВИАТУРЫ: Сохраняй аббревиатуры (АГ, ИМ, КТ, СМАД) КАК ЕСТЬ.\n"
    "4. ЕДИНИЦЫ ИЗМЕРЕНИЯ: Дописывай их к цифрам.\n"
    "5. СНОСКИ: Весь мелкий шрифт помести в массив `footnotes`. НЕ встраивай сноски внутрь узлов!"
)

PROMPT_TEXT_TABLE = (
    "Перед тобой текстовая таблица (многоколоночная или одноколоночный список критериев/чек-лист).\n"
    "ТВОЯ ЦЕЛЬ: Извлечь атомарные факты для векторного поиска (RAG).\n\n"
    "1. ГЛУБОКАЯ 2D-ИЕРАРХИЯ В 'context':\n"
    "   - Для многоколоночных таблиц: [Заголовок столбца | Глобальная категория | Название строки].\n"
    "   - Для одноколоночных таблиц и списков: выноси жирные строки-разделители и категории верхнего уровня в 'context' как иерархический путь: [Название таблицы | Глобальная категория | Подкатегория].\n\n"
    "2. ГРАНУЛЯРНОСТЬ И ВЛОЖЕННЫЕ СПИСКИ:\n"
    "   - Каждый пункт списка (•, ◦, тире) извлекай как ОТДЕЛЬНЫЙ факт, полностью дублируя родительский 'context'.\n"
    "   - Если внутри пункта указано название параметра или критерия, выноси его имя в 'context', а пороговые значения, формулы и описания — в 'value'.\n\n"
    "3. ЦЕЛОСТНОСТЬ И СКЛЕЙКА ПЕРЕНОСОВ:\n"
    "   - Склеивай многострочные предложения внутри одного пункта в один связный текст без разрывов.\n\n"
    "4. СНОСКИ И ЕДИНИЦЫ ИЗМЕРЕНИЯ:\n"
    "   - Обязательно прикрепляй единицы измерения к цифрам.\n"
    "   - Сохраняй символы сносок рядом со словами, а полный текст сносок помещай в массив footnotes."
)

PROMPTS_MAP = {
    ImageCategory.FLOWCHART: PROMPT_FLOWCHART,
    ImageCategory.MATRIX: PROMPT_MATRIX,
    ImageCategory.TEXT_TABLE: PROMPT_TEXT_TABLE
}


class ExtractedFact(BaseModel):
    context: str = Field(description="Условия, узлы или источники влияния.")
    value: str = Field(description="Итоговые значения, действия или связи.")


class ImageExtraction(BaseModel):
    analysis_status: str = Field(description="Статус: 'success' или 'failed'")
    diagram_summary: str = Field(default="", description="ТОЛЬКО для блок-схем (flowchart): цепочка шагов. Для таблиц оставить пустым.")
    source_type: str = Field(description="Тип контента (flowchart, matrix, text_table)")
    global_context: str = Field(description="Общее название изображения/таблицы")
    plain_text_blocks: list[str] = Field(default=[], description="Сплошные абзацы текста, не являющиеся схемой/таблицей.")
    facts: list[ExtractedFact] = Field(description="Массив извлеченных фактов: логические пары Условия -> Значение.")
    footnotes: list[str] = Field(default=[], description="Массив сносок внизу страницы.")


def classify_image_category(pil_img) -> ImageCategory:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    router_img = pil_img.copy()
    router_img.thumbnail((512, 512))

    config = types.GenerateContentConfig(
        system_instruction="Определи визуальный класс медицинского изображения для последующей оцифровки.",
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=ImageRouter,
    )

    model_name = "gemini-3.5-flash-lite"
    max_retries = 20
    attempt = 1

    while attempt <= max_retries:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[router_img, "Определи категорию изображения."],
                config=config
            )
            data = json.loads(response.text)
            category = ImageCategory(data.get("category", ImageCategory.TEXT_TABLE))
            print(f"🖼️ [Vision Router] Найдена картинка -> Класс: '{category.value}' (Модель: {model_name})")
            return category
        except Exception as e:
            print(f"⚠️ [Vision Router] Ошибка API ({model_name}, попытка {attempt}): {e}")
            if attempt == max_retries:
                print(f"❌ [Vision Router] Фатальная ошибка после {max_retries} попыток. Возвращаем TEXT_TABLE по умолчанию.")
                return ImageCategory.TEXT_TABLE
            print("🔄 Повтор через 5 сек...")
            time.sleep(5)
            attempt += 1


def slice_image_smart_opencv(pil_img):
    open_cv_image = np.array(pil_img)
    if open_cv_image.shape[-1] == 4:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 150 and h > 50:
            crops.append(pil_img.crop((x, y, x + w, y + h)))
    return crops if crops else [pil_img]


def describe_image(pil_img, previous_table_title=None, previous_page_text=None):
    if pil_img is None:
        return "", None, False

    def enhance_for_ocr(img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
        img = img.filter(ImageFilter.SHARPEN)
        return img

    pil_img = enhance_for_ocr(pil_img)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    category = classify_image_category(pil_img)
    is_table_img = category in [ImageCategory.TEXT_TABLE, ImageCategory.MATRIX]
    use_opencv = (category == ImageCategory.MATRIX)

    specialized_prompt = PROMPTS_MAP.get(category, PROMPT_TEXT_TABLE)

    continuation_prompt = ""
    if previous_table_title:
        continuation_prompt += f"\nВозможно, это продолжение таблицы '{previous_table_title}' с предыдущей страницы."

    if previous_page_text and is_table_img:
        continuation_prompt += (
            f"\n\n--- ЭТАЛОННЫЙ КОНТЕКСТ И ШАПКИ ПРЕДЫДУЩЕЙ СТРАНИЦЫ ---\n{previous_page_text}\n\n"
            f"ПРАВИЛА ОБРАБОТКИ РАЗРЫВА ТАБЛИЦЫ:\n"
            f"1. ШАПКА ТАБЛИЦЫ: Если на картинке продолжается таблица без заглавных столбцов, восстанови названия колонок и путь 'context' из последних блоков 'Условия: [...]' прошлой страницы.\n"
            f"2. СКЛЕЙКА ОБОРАВАННОЙ СТРОКИ: Если первая ячейка на текущей картинке является остатком фразы, оборванной внизу прошлой страницы (например, начинается со строчной буквы), ОБЯЗАТЕЛЬНО восстанови полный 'context' (активный препарат/параметр) из последней строки прошлой страницы. Категорически запрещено выдумывать абстрактные контексты вроде '[Риск и клинический эффект]'!\n"
            f"3. НОВАЯ ТАБЛИЦА: Если на картинке представлена самостоятельная новая таблица со своей шапкой — полностью проигнорируй текст прошлой страницы."
        )

    sys_instr = (
        "Ты — высокоточный медицинский дата-экстрактор клинических руководств.\n"
        "Твоя задача — извлекать клинические правила в формате: 'Условия: [Иерархический контекст] => Значение: [Факты/Показатели]'.\n\n"
        "ПРАВИЛА:\n"
        "1. ЧИСТОТА КОНТЕКСТА: 'context' должен содержать только реальные медицинские сущности (препараты, дозы, шкалы, градации, исходы).\n"
        "2. АББРЕВИАТУРЫ: Сохраняй стандартные медицинские сокращения (АГ, ДСУ, АВБ, ЧСС) в оригинальном виде.\n"
        "3. СНОСКИ И ЕДИНИЦЫ ИЗМЕРЕНИЯ (СТРОГО):\n"
        "   - Внутри текста и значений фактов НЕ используй надстрочные цифры-индексы (¹, ², ³, ⁴). Форматируй сноски через пробел и скобки: '[1]', '[2]' или '[сноска 1]'. Категорически запрещено приклеивать сноски к словам (пиши 'заболевание [1]', а не 'заболевание¹').\n"
        "   - ЕДИНИЦЫ ИЗМЕРЕНИЯ ПЛОЩАДИ: Единицы 'м²' (метры квадратные) и 'кг/м²' должны ВСЕГДА оставаться со степенью ² (квадрат). Не заменяй 'м²' на 'м³' или 'м⁴' из-за номеров сносок (пиши строго: '1,73 м² [3]', '1,73 м² [4]').\n"
        "   - Полные расшифровки сносок внизу страницы переписывай в массив footnotes целиком до последней буквы и цифры без сокращений и многоточий ('...').\n"
        "4. ПОСТОРОННИЙ ТЕКСТ И ЗАГОЛОВКИ СНИЗУ/СВЕРХУ (СТРОГИЙ ЗАПРЕТ): Если в область картинки снизу или сверху попали заголовки следующих разделов (например, 'Приложение Б...', 'Глава ...', 'Раздел ...'), названия новых таблиц или абзацы текста БЕЗ маркеров сносок — ПОЛНОСТЬЮ ПРОИГНОРИРУЙ ИХ. Не добавляй их ни в facts, ни в footnotes, ни в plain_text_blocks (их обрабатывает текстовый парсер).\n"
        "5. ЦЕЛОСТНОСТЬ ДАННЫХ: Не разрывай предложения на части. Не выдумывай и не вставляй искусственные заголовки (например, 'Продолжение таблицы') или маркеры '[текст обрывается]' внутри фактов, если их физически нет на самой картинке.\n\n"
        f"--- СПЕЦИАЛЬНЫЕ ПРАВИЛА ДЛЯ КЛАССА [{category.value.upper()}]: ---\n"
        f"{specialized_prompt}\n\n"
        f"{continuation_prompt}\n"
    )

    user_content = []
    if use_opencv:
        user_content.extend(slice_image_smart_opencv(pil_img))
        user_content.append("Оцифруй изолированные фрагменты как единое целое по правилам выше.")
    else:
        user_content.append(pil_img)
        user_content.append("Оцифруй изображение, строго следуя логике связей.")

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        max_output_tokens=42768,
        response_mime_type="application/json",
        response_schema=ImageExtraction,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )

    target_model = "gemini-3.7-flash"
    max_retries = 20
    attempt = 1

    while attempt <= max_retries:
        try:
            print(f"📊 [Vision Extractor] Оцифровка '{category.value}' (Модель: {target_model}, попытка {attempt}/{max_retries})...")
            response = client.models.generate_content(model=target_model, contents=user_content, config=config)
            data = json.loads(response.text)

            if data.get("analysis_status") == "success" and (
                    data.get("facts") or data.get("diagram_summary") or data.get("plain_text_blocks")):
                global_ctx = str(data.get("global_context") or "").strip()
                source_type = str(data.get("source_type") or "").strip()

                current_table_title = None
                if source_type.lower() in ["таблица", "table", "matrix", "text_table"]:
                    is_continuation = "продолжение" in global_ctx.lower() or len(global_ctx) < 10
                    if is_continuation and previous_table_title:
                        global_ctx = f"{previous_table_title} (Продолжение)"
                        current_table_title = previous_table_title
                    else:
                        current_table_title = global_ctx

                text_blocks = []

                if global_ctx and not any(w in global_ctx.lower() for w in ["фрагмент", "таблица (фрагмент)", "изображение"]):
                    text_blocks.append(f"### {global_ctx}")

                if data.get("plain_text_blocks"):
                    for block in data["plain_text_blocks"]:
                        text_blocks.append(block)
                    text_blocks.append("")

                if data.get("diagram_summary") and category == ImageCategory.FLOWCHART:
                    clean_summary = data['diagram_summary'].strip()
                    if clean_summary:
                        text_blocks.append(f"{clean_summary}\n")

                for fact in data.get("facts", []):
                    text_blocks.append(f"Условия: [{fact['context']}] => Значение: {fact['value']}")

                if data.get("footnotes"):
                    text_blocks.append("")
                    for note in data["footnotes"]:
                        text_blocks.append(note)

                final_text = "\n".join(text_blocks).strip()
                return final_text, current_table_title, is_table_img
            else:
                print(f"⚠️ [Vision Extractor] Ответ пуст или статус failed (попытка {attempt}).")
                if attempt == max_retries:
                    return "", None, False
                print("🔄 Повтор через 5 сек...")
                time.sleep(5)
                attempt += 1

        except Exception as e:
            print(f"⚠️ [Vision Extractor] Ошибка API ({target_model}, попытка {attempt}): {e}")
            if attempt == max_retries:
                print(f"❌ [Vision Extractor] Фатальная ошибка после {max_retries} попыток.")
                return "", None, False
            print("🔄 Повтор через 5 сек...")
            time.sleep(5)
            attempt += 1