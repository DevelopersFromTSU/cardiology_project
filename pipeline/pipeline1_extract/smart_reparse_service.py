import os
import json
import time
from pathlib import Path
from pipeline.pipeline1_extract.parser import parse_pdf_pro
from google import genai
from google.genai import types

# Высчитываем пути относительно сервиса
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = BASE_DIR / "pipeline" / "result"

def run_smart_reparse(book_path_pdf, book_name_str, page_number, use_prev, use_next, use_draft, current_draft, custom_prompt):
    context_text = ""
    if use_prev and page_number > 1:
        prev_file = RESULT_DIR / book_name_str / f"page_{page_number-1}.json"
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                context_text += f"--- ЭТАЛОННАЯ ШАПКА И КОНТЕКСТ С ПРЕДЫДУЩЕЙ СТРАНИЦЫ (СТР. {page_number-1}) ---\n{json.load(f).get('refined_text', '')}\n\n"

    if use_next:
        next_file = RESULT_DIR / book_name_str / f"page_{page_number+1}.json"
        if next_file.exists():
            with open(next_file, "r", encoding="utf-8") as f:
                context_text += f"--- КОНТЕКСТ СЛЕДУЮЩЕЙ СТРАНИЦЫ (СТР. {page_number+1}) ---\n{json.load(f).get('refined_text', '')}\n\n"

    elements = parse_pdf_pro(str(book_path_pdf), page_number, page_number)

    sys_instr = (
        "Ты — высокоточный медицинский редактор и дата-инженер.\n"
        "ТВОЯ ЗАДАЧА: Выполнить точечную замену ТОЛЬКО поврежденных строк таблицы в черновике, оставив весь остальной текст страницы в исходном виде.\n\n"
        "ПРАВИЛА ИЗМЕНЕНИЙ:\n"
        "1. НЕПРИКОСНОВЕННОСТЬ БАЗОВОГО ТЕКСТА (СТРОГО):\n"
        "   - Весь текст черновика, НЕ являющийся строками сломанной таблицы (заголовки, описания других шкал, клинические пояснения, примечания, сноски), оставь АБСОЛЮТНО НЕИЗМЕННЫМ.\n"
        "   - Запрещено перефразировать, сокращать, менять формулировки или удалять существующие абзацы.\n\n"
        "2. ИСПРАВЛЕНИЕ ТОЛЬКО РАЗРЫВА ТАБЛИЦЫ:\n"
        "   - Найди в черновике строки таблицы с потерянными названиями колонок ('Столбец 2', 'Колонка 1' и т.д.) или оборванными строками.\n"
        "   - Замени в этих строках абстрактные колонки на точные клинические названия препаратов/параметров из эталонного контекста прошлой страницы.\n"
        "   - Формат исправленных строк таблицы: 'Условия: [Глубокий 2D-путь] => Значение: [Факты/Показатели]'.\n\n"
        "3. ЗАПРЕТ НА КОЛОНТИТУЛЫ И ПАГИНАЦИЮ:\n"
        "   - Категорически запрещено выводить в конце или начале текста физические номера страниц (например, '222', '223'), колонтитулы и служебные типографские маркеры PDF.\n\n"
        "4. ПРИОРИТЕТ ВРАЧА:\n"
        "   - Если переданы дополнительные указания врача-эксперта, примени их к исправляемой таблице."
    )

    user_content = []
    if context_text:
        user_content.append(context_text)

    if use_draft and current_draft.strip():
        draft_instruction = (
            f"--- ЭТАЛОННЫЙ ЧЕРНОВИК СТРАНИЦЫ ИЗ JSON ---\n{current_draft}\n\n"
            "⚠️ ИНСТРУКЦИЯ: Сохрани весь этот текст без изменений, исправив ТОЛЬКО поврежденные строки таблицы (замени абстрактные названия столбцов на правильные препараты из прошлой страницы). "
            "Не добавляй в вывод номер страницы."
        )
        user_content.append(draft_instruction)

    for el in elements:
        user_content.append(el["content"])

    if custom_prompt:
        user_content.append(f"\n\n--- ДОПОЛНИТЕЛЬНЫЕ УКАЗАНИЯ ВРАЧА-ЭКСПЕРТА ---\n{custom_prompt}")

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        max_output_tokens=42768,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.7-flash",
                contents=user_content,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Ошибка API при перепарсе (попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise Exception(f"Все {max_retries} попыток исчерпаны. Ошибка: {e}")