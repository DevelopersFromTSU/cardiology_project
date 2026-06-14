import re
import os
import time  # <-- НОВЫЙ ИМПОРТ
import json
from dotenv import load_dotenv

from pipeline.pipeline1_extract.parser import parse_pdf_pro
from pipeline.pipeline1_extract.vision import describe_image
from pipeline.pipeline1_extract.refiner import refine_medical_chunk
from pipeline.utils.abbreviations import force_expand_abbreviations

def save_chunk_to_folder(chunk_data, filename, folder_name):
    """Создает указанную папку и сохраняет туда данные в формате JSON."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Создана папка: {folder_name}/")

    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(chunk_data, file, ensure_ascii=False, indent=4)
    print(f"✅ Результат сохранен: {file_path}")


def inject_vision_data(md_text, pil_images):
    for pil_img in pil_images:
        description = describe_image(pil_img)
        replacement = f"\n\n{description}\n\n"
        md_text = re.sub(r"!\[.*?\]\(.*?\)", replacement, md_text, count=1)
    return md_text


def run_pipeline(book_path, output_folder, start_page, end_page):
    # Теперь параметры функции строго совпадают с именами при вызове
    for current_page in range(start_page, end_page + 1):
        print(f"\n🔄 Начинаем обработку страницы {current_page}...")

        # Используем переданный book_path вместо глобального
        raw_markdown, image_paths = parse_pdf_pro(book_path, current_page, current_page)
        full_text = inject_vision_data(raw_markdown, image_paths)
        full_text_expanded = force_expand_abbreviations(full_text)

        refined_page = refine_medical_chunk(full_text_expanded)

        if refined_page:
            # Передаем динамический output_folder
            save_chunk_to_folder(refined_page, f"page_{current_page}.json", output_folder)
            print(f"✅ Страница {current_page} успешно сохранена.")

        if current_page < end_page:
            delay_minutes = 1
            print(f"⏳ Пауза {delay_minutes} мин. для сброса лимитов API Gemini...")
            time.sleep(delay_minutes * 60)


if __name__ == "__main__":
    load_dotenv()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_book_path = os.getenv("BOOK_PATH", "")
    book_path = os.path.normpath(os.path.join(BASE_DIR, raw_book_path))

    raw_result_dir = os.getenv("RESULT_DIR", "./result")
    output_folder = os.path.normpath(os.path.join(BASE_DIR, raw_result_dir))

    run_pipeline(
        book_path=book_path,
        output_folder=output_folder,
        start_page=82,
        end_page=100
    )
