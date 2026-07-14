import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image


load_dotenv()

class ImageRouter(BaseModel):
    needs_opencv: bool = Field(
        description="True, если это плотная многоколоночная таблица (матрица) с цифрами. "
                    "False, если это блок-схема, алгоритм со стрелками, круговая диаграмма, график или анатомический рисунок."
    )

class ExtractedFact(BaseModel):
    context: str = Field(
        description="Условия строки или узла алгоритма. Если значения имеют общий визуальный или смысловой признак (цвет, категория, группа), ОБЯЗАТЕЛЬНО вынеси его сюда один раз. ПРИМЕРЫ: '[Шаг: 1, Симптом: Кашель > 3 недель]', '[Возраст: 50, Курит, Общий цвет ячеек: Красный]'."
    )
    value: str = Field(
        description="Итоговые значения, диагнозы или действия. Для многоколоночных таблиц — группируй массив данных, связывая их с восстановленными заголовками. ПРИМЕРЫ ИДЕАЛА: 'Дозировки [Начальная, Максимальная]: 5 мг, 10 мг', 'Риски для стадий [I, II, III]: 15%, 20%, 30%', 'Действие: Назначить статин'. ЗАПРЕЩЕНО дублировать общие слова (цвет, риск) возле каждого элемента массива."
    )

# [НОВОЕ]: Строгая модель для ячейки строки вместо динамического dict, чтобы обойти ограничение additionalProperties
class RowStateCell(BaseModel):
    column_name: str = Field(
        description="Название колонки или заголовка (например 'Препарат', 'Показания', 'Дозировка')")
    cell_value: str = Field(description="Точный текст ячейки в этой колонке в самом низу изображения")


class ImageExtraction(BaseModel):
    analysis_status: str = Field(description="Статус: 'success' если данные успешно извлечены, иначе 'failed'")

    # [НОВОЕ ПОЛЕ - ДОЛЖНО БЫТЬ В САМОМ ВЕРХУ]:
    reasoning_step: str = Field(
        description="Твой внутренний монолог перед парсингом. Внимательно опиши, как устроена эта матрица/таблица, что означают цвета, есть ли продолжение с прошлой страницы и какие мелкие сноски есть внизу. Подумай, как правильно связать колонки, чтобы не допустить ошибок."
    )

    source_type: str = Field(description="Тип контента: 'таблица', 'алгоритм', 'график' или 'легенда'")
    global_context: str = Field(description="Общее название или суть изображения")
    last_row_state: list[RowStateCell] = Field(default=[], description="Снимок всей крайней нижней строки...")
    facts: list[ExtractedFact] = Field(description="Массив всех извлеченных атомарных данных")
    footnotes: list[str] = Field(default=[], description="Дословный список всех сносок...")


def check_image_type_with_llm(pil_img):
    """
    Быстрый зрительный фильтр. Решает, нужно ли резать картинку через OpenCV.
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    sys_instr = (
        "Ты — классификатор изображений. Твоя единственная задача — посмотреть на медицинскую иллюстрацию "
        "и решить, является ли она строгой, плотной таблицей (матрицей). "
        "Если на картинке есть ветвящиеся стрелки, логические деревья ('Да/Нет'), круги или силуэты — возвращай False."
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,  # Нам нужна максимальная предсказуемость
        response_mime_type="application/json",
        response_schema=ImageRouter,
    )

    try:
        print("🔍 LLM-роутер: оцениваю тип изображения...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_img, "Определи тип картинки для маршрутизации."],
            config=config
        )
        data = json.loads(response.text)
        return data.get("needs_opencv", False)
    except Exception as e:
        print(f"⚠️ Ошибка роутера: {e}. По умолчанию отключаем OpenCV.")
        return False

def slice_image_smart_opencv(pil_img):
    # 1. Конвертируем PIL в формат OpenCV
    open_cv_image = np.array(pil_img)
    if open_cv_image.shape[-1] == 4:  # Убираем альфа-канал, если есть
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    # 2. Инвертируем цвета: текст и цветные блоки станут белыми, а фон черным
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 3. "Размываем" пиксели, чтобы склеить цифры и слова в единые блоки
    # (50, 10) означает, что мы сильно склеиваем по горизонтали и слабо по вертикали
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 4. Ищем контуры получившихся блоков
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    # 5. Сортируем контуры сверху вниз, слева направо
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Отфильтровываем слишком мелкий мусор (пылинки, одиночные буквы)
        if w > 150 and h > 50:
            crop = pil_img.crop((x, y, x + w, y + h))
            crops.append(crop)

    # Если OpenCV ничего не нашел (что вряд ли), возвращаем исходную картинку
    return crops if crops else [pil_img]

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
        "Ты — эксперт-аналитик по оцифровке медицинских документов (таблиц, блок-схем, графиков) для векторных баз данных (RAG). "
        "Твоя задача — извлечь всю фактологию из изображения и преобразовать ее в плоский массив семантических блоков 'Условие -> Значения'.\n\n"
        "УНИВЕРСАЛЬНЫЕ ПРАВИЛА:\n"
        "1. АБСОЛЮТНАЯ ТОЧНОСТЬ (ZERO HALLUCINATION): Запрещено перефразировать клинические термины, диагнозы и дозировки. Копируй текст дословно. Особое внимание уделяй антонимам (увеличивается/уменьшается, показан/противопоказан).\n"
        "2. АТОМАРНОСТЬ ПОСТРОЧНО (КРИТИЧЕСКИ ВАЖНО): Один факт = одна логическая строка таблицы или один узел алгоритма. Если в строке 4 колонки значений, запиши все 4 значения в ОДИН факт (в поле value). Не дроби одну строку на 4 отдельных факта! Векторная база должна получать всю строку целиком.\n"
        "3. РАБОТА С ТАБЛИЦАМИ: В `context` записывай все глобальные условия и заголовок строки. В `value` перечисляй значения всех столбцов для этой строки. Если у столбцов/строк нет заголовков, нумеруй их '(столбец 1)', '(строка 2)'. Если ячейка объединена по вертикали (rowspan), дублируй ее значение для каждой логической подстроки.\n"
        "4. НИКАКИХ 'ГОЛЫХ' ЦИФР: Каждое число должно иметь объяснение. Пиши 'Уровень глюкозы: 5.5 ммоль/л', а не просто '5.5'. Для шкал риска обязательно пиши слово 'Риск' и знак '%', а также расшифровку цвета из легенды.\n"
        "5. РАБОТА С АЛГОРИТМАМИ (БЛОК-СХЕМАМИ): Если перед тобой дерево клинических решений (стрелки, блоки 'Да/Нет'), в поле 'context' записывай путь [Шаг 1 -> Условие 'Да' -> Шаг 2], а в 'value' — итоговое действие.\n"
        "6. РАСШИФРОВКА ВИЗУАЛА (ЛЕГЕНДЫ И СИМВОЛЫ): Переводи цвета (красный, желтый, зеленый), символы (+, -, ++, *, ↑, ↓) в текстовый формат. (Например: 'Зеленый цвет' -> 'Низкий риск').\n"
        "7. СНОСКИ И ИСКЛЮЧЕНИЯ: Ищи текст под звездочками (*, **), цифрами или словами 'Примечание'. Извлекай их дословно. Если текст легенды уже распарсен как факты, не дублируй его в сноски. Читай ТОЛЬКО те пиксели, которые есть на фото. ЗАПРЕЩЕНО добавлять медицинские факты из своей памяти!\n"
        "8. АНТИ-ЛЕНЬ (КРИТИЧЕСКИ ВАЖНО): Тебе СТРОГО ЗАПРЕЩЕНО сокращать таблицы, использовать многоточия или пропускать строки. Ты ОБЯЗАН извлечь абсолютно каждую строку давления/показателя. Проверь себя перед выдачей результата!"
        "9. УМНАЯ ГРУППИРОВКА (DRY - Don't Repeat Yourself): Если строка содержит несколько значений с одинаковыми атрибутами (например, все 4 ячейки красного цвета), вынеси этот атрибут в `context` один раз. В `value` сгруппируй только сами цифры. Пример плохого ответа: 'Риск 1: 10% (Красный); Риск 2: 12% (Красный)'. Пример хорошего ответа (context: '...Цвет: Красный', value: 'Значения для стадий [I, II, III]: 10%, 12%, 15%').\n"
        f"{continuation_prompt}\n\n"
    )

    # Разрезаем целевую картинку на 4 фрагмента (с нахлестом 10%)
    # Если картинка узкая или короткая, можно добавить логику проверки высоты,
    # но Gemini отлично "съест" и короткие кропы.
    image_slices = slice_image_smart_opencv(pil_img)

    use_opencv = check_image_type_with_llm(pil_img)

    user_content = []

    # 2. Формируем контент и базовую инструкцию строго в зависимости от ответа роутера
    if use_opencv:
        print("✂️ Роутер решил: Это плотная таблица. Включаем OpenCV.")
        # Режем картинку только если роутер дал добро
        image_slices = slice_image_smart_opencv(pil_img)
        user_content.extend(image_slices)

        prompt_instruction = (
            "Перед тобой изолированные смысловые фрагменты (квадранты) одной большой таблицы или схемы (они идут первыми в списке). "
            "Я разрезал изображение на логические блоки с помощью компьютерного зрения, чтобы тебе было легче фокусироваться на конкретных массивах данных. "
            "Оцифруй ИМЕННО ЭТИ фрагменты как единое целое в один общий массив фактов."
        )
    else:
        print("🖼 Роутер решил: Это сложная схема/алгоритм. Передаем целиком.")
        # Передаем картинку без нарезки
        user_content.append(pil_img)

        prompt_instruction = (
            "Перед тобой сложная медицинская схема, алгоритм или график. "
            "Я передал её целиком, чтобы ты видел(а) все логические связи и стрелки. "
            "Оцифруй этот граф в логические шаги (Условие -> Значение)."
        )

    # 3. В КОНЕЦ добавляем полноразмерную страницу для контекста
    if full_page_img is not None:
        user_content.append(full_page_img)
        prompt_instruction += (
            " Самое последнее изображение в списке — это скриншот всей страницы целиком. "
            "Используй его ИСКЛЮЧИТЕЛЬНО для лучшего кругозора, понимания структуры и общего контекста. "
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО извлекать конкретные цифры и факты из последнего изображения — извлекай их только из первых фрагментов!"
        )

    if previous_table_title or previous_row_state:
        prompt_instruction += " Учти правила переноса таблиц с прошлой страницы."

    # Добавляем итоговый текст промпта
    user_content.append(prompt_instruction)

    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.1,
        max_output_tokens=82768,
        response_mime_type="application/json",
        response_schema=ImageExtraction,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    )

    try:
        print("⏳ Оцифровка таблицы с контролем многоколоночных разрывов...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=config
        )

        # === ДОБАВЬТЕ ЭТОТ БЛОК ДЛЯ ПОДСЧЕТА ТОКЕНОВ ===
        if response.usage_metadata:
            in_tokens = response.usage_metadata.prompt_token_count
            out_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

            print("📊 СТАТИСТИКА ТОКЕНОВ:")
            print(f"   ➤ Входящие (картинка + промпт): {in_tokens}")
            print(f"   ➤ Исходящие (ответ JSON): {out_tokens}")
            print(f"   ➤ Всего потрачено: {total_tokens}")
        # ===============================================

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