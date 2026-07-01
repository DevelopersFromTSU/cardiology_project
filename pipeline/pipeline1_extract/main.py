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


def inject_vision_data(document_elements):
    final_blocks = []

    for item in document_elements:
        if item["type"] == "text":
            final_blocks.append(item["content"])
        elif item["type"] == "image":
            # [МОДИФИЦИРОВАНО]: Передаем в функцию оцифровки кроп и скриншот всей страницы
            crop_img = item["content"]
            full_page_img = item.get("full_page_image")

            description = describe_image(crop_img, full_page_img=full_page_img)
            if description.strip():
                final_blocks.append(description)

    return "\n\n".join(final_blocks)


def run_pipeline(book_path, output_folder, start_page, end_page):
    for current_page in range(start_page, end_page + 1):
        print(f"\n🔄 Начинаем обработку страницы {current_page}...")

        # 1. Парсим элементы страницы
        document_elements = parse_pdf_pro(book_path, current_page, current_page)

        page_final_blocks = []

        for item in document_elements:
            if item["type"] == "text":
                # Обычный текст прогоняем через цепочку очистки (Рефайнер)
                text_expanded = force_expand_abbreviations(item["content"])
                refined_data = refine_medical_chunk(text_expanded)

                # Извлекаем очищенный текст из ответа рефайнера
                if isinstance(refined_data, dict) and "refined_text" in refined_data:
                    if refined_data["refined_text"].strip():
                        page_final_blocks.append(refined_data["refined_text"])

            elif item["type"] == "image":
                # [ИСПРАВЛЕНО]: Текст из Vision-модели полностью защищен от повторной фильтрации!
                crop_img = item["content"]
                full_page_img = item.get("full_page_image")

                vision_description = describe_image(crop_img, full_page_img=full_page_img)
                if vision_description.strip():
                    # Только раскрываем медицинские аббревиатуры, но не переписываем текст
                    vision_description_expanded = force_expand_abbreviations(vision_description)
                    page_final_blocks.append(vision_description_expanded)

        # 2. Соединяем всё в единый чистый медицинский чанк для страницы
        combined_page_text = "\n\n".join(page_final_blocks)

        # 3. Формируем финальный JSON-объект, готовый для векторайзера
        final_json_payload = {
            "analysis_status": "success" if combined_page_text.strip() else "failed",
            "refined_text": combined_page_text
        }

        if combined_page_text.strip():
            save_chunk_to_folder(final_json_payload, f"page_{current_page}.json", output_folder)
            print(f"✅ Страница {current_page} успешно сохранена без дублирования элементов.")


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
        start_page=198,
        end_page=198
    )
