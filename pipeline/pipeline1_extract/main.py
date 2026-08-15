import re
import os
import time
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pipeline.pipeline1_extract.parser import parse_pdf_pro
from pipeline.pipeline1_extract.vision import describe_image
from pipeline.utils.abbreviations import force_expand_abbreviations
from pipeline.pipeline1_extract.refiner import refine_medical_chunk, generate_page_metadata


def save_chunk_to_folder(chunk_data, filename, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Создана папка: {folder_name}/")

    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(chunk_data, file, ensure_ascii=False, indent=4)
    print(f"✅ Результат сохранен: {file_path}")


# def inject_vision_data(document_elements):
#     final_blocks = []
#     for item in document_elements:
#         if item["type"] == "text":
#             final_blocks.append(item["content"])
#         elif item["type"] == "image":
#             crop_img = item["content"]
#             full_page_img = item.get("full_page_image")
#             # Распаковываем кортеж из 3 элементов, игнорируя память (для разовой инжекции)
#             description, _, _ = describe_image(crop_img, full_page_img=full_page_img)
#             if description.strip():
#                 final_blocks.append(description)
#
#     return "\n\n".join(final_blocks)


def run_pipeline(book_path, output_folder, start_page, end_page):
    last_table_title = None
    last_row_state = {}
    previous_page_topic = None

    SHORT_TEXT_THRESHOLD = 150  # Порог длины текста (в символах)
    # ----------------------------------------------------

    for current_page in range(start_page, end_page + 1):
        # ... начало цикла for current_page in range(start_page, end_page + 1): ...
        print(f"\n🔄 Начинаем обработку страницы {current_page}...")

        document_elements = parse_pdf_pro(book_path, current_page, current_page)
        page_final_blocks = []

        # --- НОВЫЕ ПЕРЕМЕННЫЕ ДЛЯ ТРЕКИНГА ОШИБОК ---
        page_status = "success"
        page_errors = []
        # -------------------------------------------

        for item in document_elements:
            if item["type"] == "text":
                text_expanded = force_expand_abbreviations(item["content"])

                table_match = re.search(r'(Таблица\s+[\w\.\-/]+[^\n]+)', text_expanded, re.IGNORECASE)
                if table_match:
                    last_table_title = table_match.group(1).strip()
                    last_row_state = {}

                refined_data = refine_medical_chunk(text_expanded)

                # --- НОВАЯ ЛОГИКА: ПРОВЕРКА ОШИБКИ ТЕКСТА ---
                if refined_data is None:
                    page_status = "warning"
                    page_errors.append("Сбой Gemini при обработке текста")
                elif isinstance(refined_data, dict) and "refined_text" in refined_data:
                    if refined_data["refined_text"].strip():
                        page_final_blocks.append(refined_data["refined_text"])

            elif item["type"] == "image":
                crop_img = item["content"]
                vision_description, extracted_table_title, extracted_state = describe_image(
                    crop_img,
                    previous_table_title=last_table_title,
                    previous_row_state=last_row_state
                )

                # --- НОВАЯ ЛОГИКА: ПРОВЕРКА ОШИБКИ КАРТИНКИ ---
                # Если vision вернул пустую строку, значит произошел сбой
                if not vision_description.strip():
                    page_status = "warning"
                    page_errors.append("Сбой Gemini при анализе изображения")

                if extracted_table_title:
                    last_table_title = extracted_table_title
                if extracted_state and isinstance(extracted_state, dict):
                    last_row_state = extracted_state

                if vision_description.strip():
                    expanded_vision_text = force_expand_abbreviations(vision_description)
                    page_final_blocks.append(expanded_vision_text)

        # Объединяем блоки текущей страницы...
        combined_page_text = "\n\n".join(page_final_blocks).strip()

        if not combined_page_text:
            continue

        # ... (Блок с SHORT_TEXT_THRESHOLD остается без изменений) ...

        print("⏳ Генерация метаданных страницы (тема и теги)...")
        metadata = generate_page_metadata(combined_page_text, previous_page_topic)

        # --- НОВАЯ ЛОГИКА: ПРОВЕРКА ОШИБКИ МЕТАДАННЫХ ---
        # Если метаданные вернули дефолтную заглушку из refiner.py
        if metadata.get("topic") == "Медицинские данные":
            page_status = "warning"
            page_errors.append("Сбой генерации метаданных (fallback)")

        page_topic = metadata.get("topic", "Не определена")
        page_tags = metadata.get("tags", "Нет тегов")
        previous_page_topic = page_topic

        # --- ОБНОВЛЕННЫЙ JSON ПЕЙЛОАД ---
        final_json_payload = {
            "page": current_page,
            "analysis_status": page_status,  # Динамический статус (success или warning)
            "errors": page_errors,  # Сохраняем причину сбоя в файл
            "topic": page_topic,
            "tags": page_tags,
            "refined_text": combined_page_text
        }

        save_chunk_to_folder(final_json_payload, f"page_{current_page}.json", output_folder)
        print(f"✅ Страница {current_page} успешно сохранена.")

        # --- [НОВОЕ]: Запоминаем текущую страницу для следующих итераций ---
        last_saved_payload = final_json_payload
        last_saved_page_num = current_page

