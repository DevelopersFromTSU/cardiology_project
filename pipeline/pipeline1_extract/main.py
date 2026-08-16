import re
import os
import json
from pipeline.pipeline1_extract.parser import parse_pdf_pro
from pipeline.pipeline1_extract.vision import describe_image
from pipeline.utils.abbreviations import force_expand_abbreviations
from pipeline.pipeline1_extract.refiner import refine_medical_chunk


def save_chunk_to_folder(chunk_data, filename, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Создана папка: {folder_name}/")

    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(chunk_data, file, ensure_ascii=False, indent=4)
    print(f"✅ Результат сохранен: {file_path}")


def run_pipeline(book_path, output_folder, start_page, end_page):
    last_table_title = None
    last_row_state = {}
    previous_page_topic = None

    for current_page in range(start_page, end_page + 1):
        print(f"\n🔄 Начинаем обработку страницы {current_page}...")

        document_elements = parse_pdf_pro(book_path, current_page, current_page)
        page_final_blocks = []

        page_status = "success"
        page_errors = []
        page_topic = previous_page_topic or "Не определена"
        page_tags = "Нет тегов"

        # --- [НОВЫЕ СТРОЧКИ: ГРУППИРОВКА ТЕКСТА] ---
        # Склеиваем идущие подряд абзацы в единые крупные блоки между изображениями
        grouped_elements = []
        current_text_group = []

        for el in document_elements:
            if el["type"] == "text":
                current_text_group.append(el["content"])
            elif el["type"] == "image":
                if current_text_group:
                    grouped_elements.append({"type": "text", "content": "\n\n".join(current_text_group)})
                    current_text_group = []
                grouped_elements.append(el)

        if current_text_group:
            grouped_elements.append({"type": "text", "content": "\n\n".join(current_text_group)})
        # ------------------------------------------

        # Теперь итерируемся по сгруппированным элементам
        for item in grouped_elements:
            if item["type"] == "text":
                text_expanded = force_expand_abbreviations(item["content"])

                table_match = re.search(r'(Таблица\s+[\w\.\-/]+[^\n]+)', text_expanded, re.IGNORECASE)
                if table_match:
                    last_table_title = table_match.group(1).strip()
                    last_row_state = {}

                # Передаем ВЕСЬ массив текста страницы за один вызов
                refined_data = refine_medical_chunk(text_expanded, previous_topic=previous_page_topic)

                if refined_data is None:
                    page_status = "warning"
                    page_errors.append("Сбой Gemini при обработке текста")
                elif isinstance(refined_data, dict):
                    if refined_data.get("topic") and refined_data.get("topic") != "Медицинские данные":
                        page_topic = refined_data["topic"]
                    if refined_data.get("tags") and refined_data.get("tags") != "медицина, справочник":
                        page_tags = refined_data["tags"]

                    if refined_data.get("refined_text", "").strip():
                        page_final_blocks.append(refined_data["refined_text"])

            elif item["type"] == "image":
                crop_img = item["content"]
                vision_description, extracted_table_title, extracted_state = describe_image(
                    crop_img,
                    previous_table_title=last_table_title,
                    previous_row_state=last_row_state
                )

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

        combined_page_text = "\n\n".join(page_final_blocks).strip()

        if not combined_page_text:
            continue

        if page_topic == "Не определена":
            # Ищем заголовок прямо в сгенерированном тексте
            ctx_match = re.search(r'---\s*Контекст изображения:\s*(.*?)\s*---', combined_page_text)
            if ctx_match:
                raw_title = ctx_match.group(1).strip()
                # Отрезаем технический тип в скобках на конце, например (TEXT_TABLE)
                clean_title = re.sub(r'\s*\([a-zA-Z_]+\)$', '', raw_title)

                if clean_title:
                    page_topic = clean_title

                    # Динамически собираем теги из слов длиннее 3 букв
                    title_words = [w.strip('.,()[]"\'') for w in clean_title.split()]
                    meaningful_words = [w.lower() for w in title_words if len(w) > 3]
                    if meaningful_words:
                        page_tags = ", ".join(meaningful_words[:7])
                    else:
                        page_tags = "таблица, схема, медицина"

        previous_page_topic = page_topic

        final_json_payload = {
            "page": current_page,
            "analysis_status": page_status,
            "errors": page_errors,
            "topic": page_topic,
            "tags": page_tags,
            "refined_text": combined_page_text
        }

        save_chunk_to_folder(final_json_payload, f"page_{current_page}.json", output_folder)
        print(f"✅ Страница {current_page} успешно сохранена.")
