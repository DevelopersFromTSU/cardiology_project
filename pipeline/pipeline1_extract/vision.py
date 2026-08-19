import os
import json
from enum import Enum
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import time

load_dotenv()


class ImageCategory(str, Enum):
    FLOWCHART = "flowchart"
    MATRIX = "matrix"
    TEXT_TABLE = "text_table"


class ImageRouter(BaseModel):
    category: ImageCategory = Field(
        description="Категория медицинского изображения: "
                    "'flowchart' (блок-схемы, алгоритмы со стрелками), "
                    "'matrix' (плотные цифровые сетки SCORE2), "
                    "'text_table' (классические текстовые таблицы)."
    )


PROMPT_MATRIX = (
    "Перед тобой плотная многомерная матрица (например, тепловая карта рисков, шкала стратификации, сложная сетка дозировок).\n"
    "ТВОЯ ЦЕЛЬ: Подготовить изолированные чанки данных для векторной базы (RAG-системы). База ищет информацию по точным пересечениям параметров, поэтому точность координат критична.\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. ТОЧНЫЕ КООРДИНАТЫ (ОСЬ Y + ОСЬ X): В 'context' собери путь по оси Y (боковые категории), используя короткие "
    "напечатанные аббревиатуры (например, пиши 'КК 30-50', а не 'Клиренс креатинина'). В 'value' ОБЯЗАТЕЛЬНО привяжи "
    "цифры к их колонкам (ось X). Если точных заголовков у колонок нет, КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО их додумывать! В "
    "таком случае укажи общий видимый заголовок перед массивом (Пример 'value': '[Название группы/оси]: 27%, 28%, "
    "30% (Цвет фона)'). ВНИМАНИЕ: Обязательно ищи вертикально напечатанные заголовки осей. Если над колонкой цифр написано другое слово (например, название самой таблицы), не путай его с осью!\n"
    "2. ГОРИЗОНТАЛЬНАЯ ГРУППИРОВКА (ОПТИМИЗАЦИЯ): Чтобы не перегружать ответ, КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО создавать объект для каждой ячейки отдельно. Собери весь ряд значений по горизонтали в одно поле 'value' с соблюдением осей из пункта 1.\n"
    "3. ИЗОЛЯЦИЯ: Никогда не смешивай разные глобальные блоки (например, разные гендерные вкладки, возрастные группы или стадии болезни) в одном факте. Каждая логическая строка матрицы — это один независимый объект JSON.\n"
    "4. ПУСТЫЕ ЗОНЫ: Игнорируй пустые ячейки и графический мусор.\n"
    "5. ЕДИНИЦЫ ИЗМЕРЕНИЯ (ИСКЛЮЧЕНИЕ ИЗ ПРАВИЛА): Если значения представлены 'голыми' цифрами, но из контекста (легенды, оси, заголовки, сноски) понятно их измерение (например, %, мг, ммоль/л), ОБЯЗАТЕЛЬНО прикрепляй эти единицы к цифрам (например, пиши '27%', а не '27')."
)

PROMPT_FLOWCHART = (
    "Перед тобой медицинское визуальное представление (клинический алгоритм, описательная инфографика, матрица, патогенетический цикл или анатомическая схема).\n"
    "ТВОЯ ЦЕЛЬ: Оцифровать данные для RAG-системы. Векторная база ищет по парам 'Ключ (context) -> Значение (value)'. Твоя задача — быть точным аналитиком, а не пересказывать суть своими словами.\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. АДАПТИВНОЕ ИЗВЛЕЧЕНИЕ СВЯЗЕЙ:\n"
    "   - Алгоритмы: В 'context' пиши [Родитель -> Условие перехода], в 'value' — [Целевой блок/Действие].\n"
    "   - Описания/Анатомия: В 'context' — [Название структуры], в 'value' — [Описание/Функция].\n"
    "   - Матрицы/Графики: В 'context' — [Ось X, Ось Y], в 'value' — [Показатель].\n"
    "   - Циклы: В 'context' — [Текущий этап], в 'value' — [Следующий этап].\n"
    "2. ОПТИМИЗАЦИЯ И ЦЕЛОСТНОСТЬ: Объединяй общие условия. Не дроби списки препаратов/симптомов в целевом блоке — пиши всё в одно поле 'value'. Если блоки имеют общий фон — добавляй его в 'context'.\n"
    "3. АББРЕВИАТУРЫ И ГРАММАТИКА (СТРОГО!): Сохраняй аббревиатуры (АГ, ИМ, КТ, СМАД) КАК ЕСТЬ. Категорически запрещено расшифровывать их в лоб, если это ломает русский язык (например, нельзя писать 'лечение резистентной артериальная гипертензия (АГ)').\n"
    "4. ЕДИНИЦЫ ИЗМЕРЕНИЯ: Если видишь голые цифры, но из схемы понятно измерение (%, мг), обязательно дописывай его к цифре.\n"
    "5. ИЗОЛИРОВАННЫЙ ТЕКСТ И СНОСКИ (КРИТИЧНО): НЕ ПЫТАЙСЯ встраивать сноски (¹, ²) внутрь узлов! Весь мелкий шрифт, "
    "расшифровки под звездочками и сопроводительные плавающие абзацы (например, демографические данные) скопируй 'как "
    "есть' и помести в массив `footnotes`. Если массива нет, добавь их в конец `diagram_summary`. Ни одно слово с "
    "картинки не должно быть потеряно! КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать 'легенду' или самостоятельно расшифровывать "
    "сноски (¹, ², ³), основываясь на названиях блоков!"
)

PROMPT_TEXT_TABLE = (
    "Перед тобой текстовая таблица сложной структуры.\n"
    "ТВОЯ ЦЕЛЬ: Извлечь факты для векторного поиска (RAG). Базе нужны короткие, точные факты с полным контекстом, а не длинные простыни текста.\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. ГЛУБОКАЯ 2D-ИЕРАРХИЯ (ВЕРТИКАЛЬ + ГОРИЗОНТАЛЬ): В поле 'context' собери АБСОЛЮТНО ВЕСЬ иерархический путь ячейки. Сканируй структуру вверх и влево! Если над группой параметров есть промежуточная объединяющая строка-подзаголовок (например, категория, синдром или диагноз, к которому относятся все строки ниже), КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ее терять. Полный путь должен выглядеть так: [Главный заголовок столбца | Глобальная категория | Промежуточный подзаголовок (если есть) | Название конкретной строки].\n"
    "2. ГРАНУЛЯРНОСТЬ ДАННЫХ (ОБЯЗАТЕЛЬНО): Если в ячейке находится маркированный список (через тире, точки, цифры или просто абзацы), КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО сливать его в один длинный текст! Разбей каждый пункт списка на отдельный, независимый факт (отдельный объект JSON). При этом для каждого такого факта ПОЛНОСТЬЮ дублируй родительский 'context'.\n"
    "3. ЦЕЛОСТНОСТЬ ПРЕДЛОЖЕНИЙ: Извлекай текст самого факта целиком, не обрезая слова.\n"
    "4. СНОСКИ: Сохраняй символы сносок рядом со словом.\n"
    "5. ЕДИНИЦЫ ИЗМЕРЕНИЯ (ИСКЛЮЧЕНИЕ ИЗ ПРАВИЛА): Если значения представлены 'голыми' цифрами, но из контекста (легенды, оси, заголовки, сноски) понятно их измерение (например, %, мг, ммоль/л), ОБЯЗАТЕЛЬНО прикрепляй эти единицы к цифрам (например, пиши '27%', а не '27')."
)


PROMPTS_MAP = {
    ImageCategory.FLOWCHART: PROMPT_FLOWCHART,
    ImageCategory.MATRIX: PROMPT_MATRIX,
    ImageCategory.TEXT_TABLE: PROMPT_TEXT_TABLE
}


class ExtractedFact(BaseModel):
    context: str = Field(description="Условия, узлы или источники влияния.")
    value: str = Field(description="Итоговые значения, действия или связи.")


class RowStateCell(BaseModel):
    column_name: str = Field(description="Название колонки.")
    cell_value: str = Field(description="Точный текст ячейки.")


class ImageExtraction(BaseModel):
    analysis_status: str = Field(description="Статус: 'success' или 'failed'")
    diagram_summary: str = Field(default="", description="Описание схем и графиков.")
    source_type: str = Field(description="Тип контента")
    global_context: str = Field(description="Общее название изображения")
    last_row_state: list[RowStateCell] = Field(default=[])
    plain_text_blocks: list[str] = Field(
        default=[],
        description="Сплошные абзацы текста (например, введения или выводы), не являющиеся схемой. Копировать СЛОВО В СЛОВО."
    )
    facts: list[ExtractedFact] = Field(
        description="Массив извлеченных данных ТОЛЬКО из таблиц и блок-схем: строго логические связи."
    )
    footnotes: list[str] = Field(
        default=[],
        description="Массив всех сносок внизу страницы."
    )


def classify_image_category(pil_img, max_retries=3) -> ImageCategory:
    """Классификатор типа изображения с легкой моделью."""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    router_img = pil_img.copy()
    router_img.thumbnail((512, 512))

    sys_instr = (
        "Ты — эксперт-классификатор медицинских документов. Твоя единственная задача — посмотреть "
        "на изображение и определить его визуальный класс для последующей оцифровки."
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=ImageRouter,
    )

    for attempt in range(max_retries):
        try:
            print(f"🔍 LLM-роутер: классифицирую изображение (попытка {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[router_img, "Определи категорию изображения."],
                config=config
            )

            if response.usage_metadata:
                print(f"📊 [Роутер] Токены -> Вход: {response.usage_metadata.prompt_token_count} | "
                      f"Выход: {response.usage_metadata.candidates_token_count} | "
                      f"Всего: {response.usage_metadata.total_token_count}")

            data = json.loads(response.text)
            return ImageCategory(data.get("category", ImageCategory.TEXT_TABLE))

        except Exception as e:
            print(f"⚠️ Ошибка роутера (попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("🔄 Ждем 5 секунд и повторяем запрос к API...")
                time.sleep(5)
            else:
                print("❌ Все попытки исчерпаны. Откат на класс по умолчанию (text_table).")
                return ImageCategory.TEXT_TABLE


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
            crop = pil_img.crop((x, y, x + w, y + h))
            crops.append(crop)
    return crops if crops else [pil_img]


def describe_image(pil_img, previous_table_title=None, previous_row_state=None, max_retries=10):
    if pil_img is None:
        return "", None, {}

    def enhance_for_ocr(img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
        img = img.filter(ImageFilter.SHARPEN)
        return img

    pil_img = enhance_for_ocr(pil_img)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    category = classify_image_category(pil_img)
    print(f"🎯 Роутер определил категорию: {category.value}")

    use_opencv = (category == ImageCategory.MATRIX)
    specialized_prompt = PROMPTS_MAP.get(category, PROMPT_TEXT_TABLE)

    continuation_prompt = ""
    if previous_table_title:
        continuation_prompt += f"\nВНИМАНИЕ: Это изображение — продолжение таблицы '{previous_table_title}' с предыдущей страницы."

    if previous_row_state and isinstance(previous_row_state, dict) and len(previous_row_state) > 0:
        headers_list = list(previous_row_state.keys())
        state_json_str = json.dumps(previous_row_state, ensure_ascii=False)

        continuation_prompt += (
            f"\nКРИТИЧЕСКОЕ ПРАВИЛО ПЕРЕНОСА (РАЗРЫВ ТАБЛИЦЫ):\n"
            f"1. ШАПКА ТАБЛИЦЫ: На этой картинке нет заглавных столбцов. Ты ОБЯЗАН использовать следующие названия столбцов из прошлой страницы: {headers_list}.\n"
            f"2. СТРОГАЯ ПРИВЯЗКА: Мысленно наложи эти столбцы слева направо на текущую картинку и формируй условия (context) строго по ним.\n"
            f"3. СКЛЕЙКА СТРОК: Вот последняя строка прошлой страницы: {state_json_str}. Если первая строка на этой картинке оборвана и выглядит как логическое завершение прошлой — объедини их."
        )

    sys_instr = (
        "Ты — бездушный OCR-парсер и координатный экстрактор. Твоя единственная задача — перенести текст с картинки в JSON с абсолютной, буквальной точностью.\n\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. СТРУКТУРА ТЕКСТА: Если видишь обычный абзац текста (не схему) — помести его целиком в 'plain_text_blocks', не пытайся разбивать его на условия и значения!\n"
        "2. КАТЕГОРИЧЕСКИЙ ЗАПРЕЩЕНО: Не переводи, не интерпретируй и не расшифровывай аббревиатуры. Пиши строго как в оригинале (например, 'АГ', 'СМАД', 'ИМ').\n"
        "3. СОХРАНЕНИЕ ИНДЕКСОВ: Сноски (¹, ², ³) переноси вместе со словами без пробелов.\n"
        "4. ЧИСТОТА КОНТЕКСТА: Не дублируй главное название таблицы (оно уже в 'global_context') в каждый 'context' факта.\n"
        "5. ОПИСАНИЕ СХЕМЫ (diagram_summary): Если схема требует текстового описания структуры, напиши его, опираясь ТОЛЬКО на видимые термины. Не придумывай патогенез или алгоритмы, если их нет на картинке.\n\n"
        f"--- СПЕЦИАЛЬНЫЕ ПРАВИЛА ДЛЯ КЛАССА [{category.value.upper()}]: ---\n"
        f"{specialized_prompt}\n\n"
        f"{continuation_prompt}\n"
    )

    user_content = []
    if use_opencv:
        print("✂️ Включаем OpenCV-нарезку для табличного контента.")
        image_slices = slice_image_smart_opencv(pil_img)
        user_content.extend(image_slices)
        prompt_instruction = "Оцифруй изолированные фрагменты как единое целое по правилам выше."
    else:
        print("🖼 Передаем изображение целиком для сохранения связей.")
        user_content.append(pil_img)
        prompt_instruction = "Оцифруй схему целиком, строго следуя логике связей."

    user_content.append(prompt_instruction)

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        max_output_tokens=42768,
        response_mime_type="application/json",
        response_schema=ImageExtraction,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )

    # Базовая модель по умолчанию в зависимости от категории изображения
    base_model = "gemini-3.5-flash-lite" if category == ImageCategory.TEXT_TABLE else "gemini-3.7-flash"

    for attempt in range(max_retries):
        try:
            # Если легкая модель вернула пустой результат или упала с 1-й попытки — эскалируем на gemini-3.7-flash
            current_model = "gemini-3.7-flash" if attempt > 0 else base_model
            if attempt > 0 and current_model != base_model:
                print(f"⚡ Эскалация: переключаем попытку {attempt + 1} на тяжелую модель {current_model}...")

            response = client.models.generate_content(
                model=current_model,
                contents=user_content,
                config=config
            )

            if response.usage_metadata:
                print(f"📊 [Анализатор] Токены -> Вход: {response.usage_metadata.prompt_token_count} | "
                      f"Выход: {response.usage_metadata.candidates_token_count} | "
                      f"Всего: {response.usage_metadata.total_token_count}")

            data = json.loads(response.text)

            if data.get("analysis_status") == "success" and (data.get("facts") or data.get("diagram_summary") or data.get("plain_text_blocks")):
                global_ctx = str(data.get("global_context") or "").strip()
                source_type = str(data.get("source_type") or "").strip()

                raw_row_list = data.get("last_row_state", [])
                row_state = {}
                if isinstance(raw_row_list, list):
                    for cell in raw_row_list:
                        if isinstance(cell, dict) and "column_name" in cell and "cell_value" in cell:
                            row_state[cell["column_name"]] = cell["cell_value"]

                non_empty_vals = [str(val).strip() for val in row_state.values() if str(val).strip()]
                if non_empty_vals and all(val.endswith((".", "!", "?", ";")) for val in non_empty_vals):
                    row_state = {}

                current_table_title = None
                if source_type.lower() in ["таблица", "table", "matrix", "text_table"]:
                    is_continuation = "продолжение" in global_ctx.lower() or len(global_ctx) < 10
                    if is_continuation and previous_table_title:
                        global_ctx = f"{previous_table_title} (Продолжение)"
                        current_table_title = previous_table_title
                    else:
                        current_table_title = global_ctx

                text_blocks = [f"--- Контекст изображения: {global_ctx} ({source_type}) ---"]

                if data.get("plain_text_blocks"):
                    for block in data["plain_text_blocks"]:
                        text_blocks.append(block)
                    text_blocks.append("")

                if data.get("diagram_summary"):
                    text_blocks.append(f"Механизмы и связи (описание схемы): {data['diagram_summary']}\n")

                for fact in data.get("facts", []):
                    text_blocks.append(f"Условия: [{fact['context']}] => Значение: {fact['value']}")

                if data.get("footnotes"):
                    text_blocks.append("")
                    for note in data["footnotes"]:
                        text_blocks.append(note)

                final_text = "\n".join(text_blocks)
                return final_text, current_table_title, row_state
            else:
                print(f"⚠️ Ответ модели пустой или status != success (попытка {attempt + 1}/{max_retries})")

        except Exception as e:
            print(f"⚠️ Ошибка Gemini в экстракторе (попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("🔄 Ждем 5 секунд перед повторным запросом...")
                time.sleep(5)
            else:
                print("❌ Все попытки распознавания картинки исчерпаны.")

    return "", None, {}