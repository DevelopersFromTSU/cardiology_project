import os
import json
import time
from pathlib import Path
from pipeline.pipeline1_extract.parser import parse_pdf_pro
from google import genai
from google.genai import types
from PIL import ImageEnhance, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = BASE_DIR / "pipeline" / "result"


def run_smart_reparse(
        book_path_pdf,
        book_name_str,
        page_number,
        prev_pages: list[int] | None = None,
        use_next: bool = False,
        use_draft: bool = True,
        current_draft: str = "",
        custom_prompt: str = ""
):
    context_text = ""

    if prev_pages:
        for p_num in sorted(prev_pages):
            prev_file = RESULT_DIR / book_name_str / f"page_{p_num}.json"
            if prev_file.exists():
                with open(prev_file, "r", encoding="utf-8") as f:
                    page_data = json.load(f)
                    context_text += (
                        f"--- ЭТАЛОННЫЙ КОНТЕКСТ И ШАПКА С ПРЕДЫДУЩЕЙ СТРАНИЦЫ (СТР. {p_num}) ---\n"
                        f"{page_data.get('refined_text', '')}\n\n"
                    )

    if use_next:
        next_file = RESULT_DIR / book_name_str / f"page_{page_number + 1}.json"
        if next_file.exists():
            with open(next_file, "r", encoding="utf-8") as f:
                context_text += f"--- КОНТЕКСТ СЛЕДУЮЩЕЙ СТРАНИЦЫ (СТР. {page_number + 1}) ---\n{json.load(f).get('refined_text', '')}\n\n"

    elements = parse_pdf_pro(str(book_path_pdf), page_number, page_number)

    sys_instr = (
        "Ты — медицинский редактор и дата-инженер.\n"
        "ТВОЯ ЗАДАЧА: Сохранить исходную структуру и формат обработки страницы ('Условия: [...] => Значение: [...]', заголовки, сплошной текст), исправив ошибки, недочеты и обрывы.\n\n"
        "ПРАВИЛА ИСПРАВЛЕНИЯ:\n"
        "1. ОБРЫВЫ ТАБЛИЦ: Восстанови потерянные названия колонок и склей разорванные строки/ячейки в единые факты, используя контекст предыдущих страниц.\n"
        "2. СНОСКИ И ПРИМЕЧАНИЯ: Перепиши весь мелкий шрифт и легенды внизу страницы полностью до последнего слова, без многоточий ('...') и сокращений.\n"
        "3. ОШИБКИ И ОПЕЧАТКИ: Исправь искаженные термины, цифры и склеенные слова, сверяясь с прикрепленными изображениями PDF.\n"
        "4. ЧИСТОТА: Не выводи номера страниц, колонтитулы и технические маркеры.\n"
        "5. ПРИОРИТЕТ ВРАЧА: Указания врача-эксперта имеют наивысший приоритет."
    )

    user_content = []
    if context_text:
        user_content.append(context_text)

    if use_draft and current_draft.strip():
        draft_instruction = (
            f"--- ТЕКУЩИЙ ЧЕРНОВИК СТРАНИЦЫ ИЗ БАЗЫ ---\n{current_draft}\n\n"
            "⚠️ ИНСТРУКЦИЯ: Сохрани структуру и оформление этого черновика. "
            "Сверяясь с картинками PDF ниже, исправь обрывы таблиц, допиши оборванные сноски и устрани ошибки/недочеты."
        )
        user_content.append(draft_instruction)

    for el in elements:
        if el["type"] == "image":
            img = el["content"]
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.8).filter(ImageFilter.SHARPEN)
            user_content.append(img)
        else:
            # Убран баг с дублированием append для текста
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

    target_model = "gemini-3.7-flash"
    max_retries = 20
    attempt = 1

    while attempt <= max_retries:
        try:
            print(f"🛠️ [Smart Reparse] Запуск исправления страницы (Модель: {target_model}, попытка {attempt}/{max_retries})...")
            response = client.models.generate_content(
                model=target_model,
                contents=user_content,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ [Smart Reparse] Ошибка API ({target_model}, попытка {attempt}): {e}")
            if attempt == max_retries:
                # В ручном режиме репарса лучше выбросить явную ошибку в UI Streamlit
                raise Exception(f"Фатальная ошибка API при перепарсе после {max_retries} попыток: {e}")
            print("🔄 Повтор через 5 сек...")
            time.sleep(5)
            attempt += 1