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

load_dotenv()


# [НОВОЕ]: Категории изображений для точечной маршрутизации
class ImageCategory(str, Enum):
    FLOWCHART = "flowchart"
    MATRIX = "matrix"
    ANATOMY = "anatomy"
    TEXT_TABLE = "text_table"


# [ИСПРАВЛЕНО]: Теперь роутер возвращает конкретный класс вместо бездушного True/False
class ImageRouter(BaseModel):
    category: ImageCategory = Field(
        description="Категория медицинского изображения: "
                    "'flowchart' (блок-схемы, алгоритмы со стрелками), "
                    "'matrix' (плотные цифровые сетки SCORE2), "
                    "'anatomy' (рисунки тела/органов с выносками), "
                    "'text_table' (классические текстовые таблицы)."
    )


PROMPT_MATRIX = (
    "Перед тобой плотная многомерная матрица (например, тепловая карта рисков, шкала стратификации, сложная сетка дозировок).\n"
    "ТВОЯ ЦЕЛЬ: Подготовить изолированные чанки данных для векторной базы (RAG-системы). База ищет информацию по точным пересечениям параметров, поэтому точность координат критична.\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. ТОЧНЫЕ КООРДИНАТЫ (ОСЬ Y + ОСЬ X): В 'context' собери путь по оси Y (боковые категории), используя короткие напечатанные аббревиатуры (например, пиши 'КК 30-50', а не 'Клиренс креатинина'). В 'value' ОБЯЗАТЕЛЬНО привяжи цифры к их колонкам (ось X). Если точных заголовков у колонок нет, КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО их додумывать! В таком случае укажи общий видимый заголовок перед массивом (Пример 'value': '[Название группы/оси]: 27%, 28%, 30% (Цвет фона)').\n"
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
    "5. ИЗОЛИРОВАННЫЙ ТЕКСТ И СНОСКИ (КРИТИЧНО): НЕ ПЫТАЙСЯ встраивать сноски (¹, ²) внутрь узлов! Весь мелкий шрифт, расшифровки под звездочками и сопроводительные плавающие абзацы (например, демографические данные) скопируй 'как есть' и помести в массив `footnotes`. Если массива нет, добавь их в конец `diagram_summary`. Ни одно слово с картинки не должно быть потеряно!"
)

PROMPT_TEXT_TABLE = (
    "Перед тобой текстовая таблица сложной структуры.\n"
    "ТВОЯ ЦЕЛЬ: Извлечь семантически плотные блоки для векторного поиска (RAG). Базе нужны законченные мысли с полным контекстом.\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. ИЕРАРХИЯ СТРОК И СТОЛБЦОВ: В поле 'context' собери путь ячейки (Заголовок столбца + Заголовок строки).\n"
    "2. ОПТИМИЗАЦИЯ И ГРУППИРОВКА: Если в ячейке находится маркированный список рекомендаций или симптомов, относящийся к одному 'context', КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО дробить его на отдельные факты. Сохрани весь список единым связным текстом внутри одного поля 'value'.\n"
    "3. ЦЕЛОСТНОСТЬ ПРЕДЛОЖЕНИЙ: Извлекай текст целиком, не обрезая абзацы.\n"
    "4. СНОСКИ: Сохраняй символы сносок рядом со словом.\n"
    "5. ЕДИНИЦЫ ИЗМЕРЕНИЯ (ИСКЛЮЧЕНИЕ ИЗ ПРАВИЛА): Если значения представлены 'голыми' цифрами, но из контекста (легенды, оси, заголовки, сноски) понятно их измерение (например, %, мг, ммоль/л), ОБЯЗАТЕЛЬНО прикрепляй эти единицы к цифрам (например, пиши '27%', а не '27')."
)

PROMPT_ANATOMY = (
    "Перед тобой анатомическая или патогенетическая схема (структуры, органы, циклы).\n"
    "ТВОЯ ЦЕЛЬ: Сформировать базу знаний для RAG-системы. Тебе нужно переложить визуальные выноски в четкие текстовые связи.\n"
    "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
    "1. КОНТЕКСТ: В 'context' укажи название структуры/органа и характер связи ('стимулирует', 'ингибирует').\n"
    "2. ОПТИМИЗАЦИЯ И ГРУППИРОВКА: Если от одного органа идет несколько выносок, описывающих один процесс, объедини их в один факт. Если в выноске список симптомов — пиши их все в одно поле 'value', не дроби.\n"
    "3. ИЕРАРХИЯ: Включай глобальные зоны (например, 'Факторы среды') в 'context'.\n"
    "4. ЕДИНИЦЫ ИЗМЕРЕНИЯ (ИСКЛЮЧЕНИЕ ИЗ ПРАВИЛА): Если значения представлены 'голыми' цифрами, но из контекста (легенды, оси, заголовки, сноски) понятно их измерение (например, %, мг, ммоль/л), ОБЯЗАТЕЛЬНО прикрепляй эти единицы к цифрам (например, пиши '27%', а не '27')."
)


PROMPTS_MAP = {
    ImageCategory.FLOWCHART: PROMPT_FLOWCHART,
    ImageCategory.MATRIX: PROMPT_MATRIX,
    ImageCategory.ANATOMY: PROMPT_ANATOMY,
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
    # reasoning_step: str = Field(description="Внутренний монолог...")
    diagram_summary: str = Field(default="", description="Описание схем и графиков.")
    source_type: str = Field(description="Тип контента")
    global_context: str = Field(description="Общее название изображения")
    last_row_state: list[RowStateCell] = Field(default=[])
    # Слегка уточняем описание, чтобы модель не пихала сюда сноски
    facts: list[ExtractedFact] = Field(
        description="Массив извлеченных данных: строго логические связи, условия и действия.")

    # [ВОТ ГЛАВНОЕ ИЗМЕНЕНИЕ: Раскомментируйте и дайте четкую инструкцию]
    footnotes: list[str] = Field(
        default=[],
        description="Массив всех сносок (текст под цифрами/звездочками) и любого плавающего изолированного текста с картинки (демография, статистика), который не вписывается в пары условие-значение."
    )


# [ИСПРАВЛЕНО]: Легкая классификация картинки перед тяжелым разбором
def classify_image_category(pil_img) -> ImageCategory:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

    try:
        print("🔍 LLM-роутер: классифицирую изображение...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_img, "Определи категорию изображения."],
            config=config
        )
        # [НОВОЕ]: Логирование токенов роутера
        if response.usage_metadata:
            print(f"📊 [Роутер] Токены -> Вход: {response.usage_metadata.prompt_token_count} | "
                  f"Выход: {response.usage_metadata.candidates_token_count} | "
                  f"Всего: {response.usage_metadata.total_token_count}")
        data = json.loads(response.text)
        return ImageCategory(data.get("category", ImageCategory.TEXT_TABLE))
    except Exception as e:
        print(f"⚠️ Ошибка роутера: {e}. Откат на класс по умолчанию (text_table).")
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


def describe_image(pil_img, previous_table_title=None, previous_row_state=None):
    if pil_img is None:
        return "", None, {}

    def enhance_for_ocr(img):
        # 1. Повышаем контрастность (делаем блеклые серые буквы жестко черными)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)  # Сильный контраст

        # 2. Убираем "мыло" от растягивания (добавляем программную резкость)
        img = img.filter(ImageFilter.SHARPEN)
        return img

    # Применяем фильтр к картинке, которая уже пришла увеличенной от parser.py
    pil_img = enhance_for_ocr(pil_img)

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # [НОВЫЕ СТРОЧКИ]: Определяем категорию ОДНИМ запросом
    category = classify_image_category(pil_img)
    print(f"🎯 Роутер определил категорию: {category.value}")

    # [НОВЫЕ СТРОЧКИ]: Автоматически решаем, нужен ли OpenCV
    # Плотные таблицы (SCORE2) и обычные текстовые таблицы режем; схемы и анатомию — передаем целиком
    use_opencv = category in [ImageCategory.MATRIX, ImageCategory.TEXT_TABLE]

    specialized_prompt = PROMPTS_MAP.get(category, PROMPT_TEXT_TABLE)

    # === ДОБАВЬТЕ ВОТ ЭТИ ДВЕ СТРОКИ ===
    print(f"📝 АКТИВНЫЙ ПРОМПТ: Выбран {category.value}")
    print(f"Текст промпта (первые 200 символов): {specialized_prompt[:200]}...")
    # ===================================

    continuation_prompt = ""

    if previous_table_title:
        continuation_prompt += f"\nВНИМАНИЕ: Это изображение может быть продолжением таблицы '{previous_table_title}'."

    if previous_row_state and isinstance(previous_row_state, dict) and len(previous_row_state) > 0:
        state_json_str = json.dumps(previous_row_state, ensure_ascii=False)
        continuation_prompt += (
            f"\nКРИТИЧЕСКОЕ ПРАВИЛО ПЕРЕНОСА: Вот окончание прошлой страницы: {state_json_str}. "
            f"Бесшовно объедини данные, если ячейки на этой странице являются продолжением."
        )

    sys_instr = (
        "Ты — строгий технический аналитик. Переведи визуальные данные в точный JSON.\n\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. АБСОЛЮТНАЯ ПОЛНОТА: Не пропусти ни одного напечатанного символа (аббревиатуры, мелкий шрифт, отдельные слова в кружках).\n"
        "2. СТРОГИЙ ЗАПРЕТ НА ГАЛЛЮЦИНАЦИИ: Извлекай ТОЛЬКО то, что напечатано. Не расшифровывай аббревиатуры самовольно.\n"
        "3. ЧИСТОТА КОНТЕКСТА (ЗАПРЕТ НА СПАМ): КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО дублировать главное название таблицы или схемы (например, 'Таблица П23/АЗ...') внутрь поля 'context' для каждого отдельного факта. Название схемы уже есть в 'global_context'. Внутри массива 'facts' используй только локальные родительские узлы (например, 'Факторы среды' или 'Регуляция АД').\n"
        "4. ОПИСАНИЕ СХЕМЫ (diagram_summary): Если схема требует текстового описания структуры, напиши его, опираясь ТОЛЬКО на видимые термины. Не придумывай патогенез или алгоритмы, если их нет на картинке.\n\n"        
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

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=user_content,
            config=config
        )

        # [НОВОЕ]: Логирование токенов роутера
        # [НОВОЕ]: Логирование токенов анализатора
        if response.usage_metadata:
            print(f"📊 [Анализатор] Токены -> Вход: {response.usage_metadata.prompt_token_count} | "
                  f"Выход: {response.usage_metadata.candidates_token_count} | "
                  f"Всего: {response.usage_metadata.total_token_count}")

        data = json.loads(response.text)

        if data.get("analysis_status") == "success" and (data.get("facts") or data.get("diagram_summary")):
            global_ctx = data.get("global_context", "").strip()
            source_type = data.get("source_type", "")

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
            if source_type == "таблица":
                is_continuation = "продолжение" in global_ctx.lower() or len(global_ctx) < 10
                if is_continuation and previous_table_title:
                    global_ctx = f"{previous_table_title} (Продолжение)"
                    current_table_title = previous_table_title
                else:
                    current_table_title = global_ctx

            text_blocks = [f"--- Контекст изображения: {global_ctx} ({source_type}) ---"]

            if data.get("diagram_summary"):
                text_blocks.append(f"Механизмы и связи (описание схемы): {data['diagram_summary']}\n")

            for fact in data.get("facts", []):
                text_blocks.append(f"Условия: [{fact['context']}] => Значение: {fact['value']}")

            if data.get("footnotes"):
                text_blocks.append("\n--- Сноски и примечания ---")
                for note in data["footnotes"]:
                    text_blocks.append(f"• {note}")

            final_text = "\n".join(text_blocks)
            return final_text, current_table_title, row_state
        else:
            return "", None, {}

    except Exception as e:
        print(f"❌ Ошибка при обработке картинки: {e}")
        return "", None, {}