import re
import os
import json
from pipeline.pipeline1_extract.parser import parse_pdf_pro
from pipeline.pipeline1_extract.vision import describe_image
from pipeline.pipeline1_extract.refiner import refine_medical_chunk, extract_page_metadata

try:
    from pipeline.utils.abbreviations import force_expand_abbreviations
except ImportError:
    def force_expand_abbreviations(text: str) -> str:
        return text


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

    for current_page in range(start_page, end_page + 1):
        print(f"\n🔄 Начинаем обработку страницы {current_page}...")

        # --- [НОВЫЙ БЛОК]: Чтение контекста из уже сохраненного JSON предыдущей страницы ---
        prev_json_path = os.path.join(output_folder, f"page_{current_page - 1}.json")
        previous_page_context = None
        previous_page_topic = None

        if os.path.exists(prev_json_path):
            try:
                with open(prev_json_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                    prev_topic = prev_data.get("topic", "")
                    prev_text = prev_data.get("refined_text", "")

                    if prev_topic and prev_topic not in ["Не определена", "Медицинские данные"]:
                        previous_page_topic = prev_topic

                    if prev_text:
                        previous_page_context = f"--- ЭТАЛОННЫЙ КОНТЕКСТ С ПРЕДЫДУЩЕЙ СТРАНИЦЫ (СТР. {current_page - 1}) ---\nТема: {prev_topic}\n\n{prev_text}"
            except Exception as e:
                print(f"⚠️ Предупреждение при чтении JSON прошлой страницы: {e}")
        # ---------------------------------------------------------------------------------

        document_elements = parse_pdf_pro(book_path, current_page, current_page)
        page_final_blocks = []
        page_status = "success"
        page_errors = []

        # 1. Группировка смежных текстовых блоков
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

        # 2. Обработка блоков
        for item in grouped_elements:
            if item["type"] == "text":
                raw_text = force_expand_abbreviations(item["content"])

                table_match = re.search(r'(Таблица\s+[\w.\-/]+[^\n]+)', raw_text, re.IGNORECASE)
                if table_match:
                    last_table_title = table_match.group(1).strip()

                refined_text = refine_medical_chunk(raw_text)
                if refined_text:
                    page_final_blocks.append(refined_text)
                else:
                    page_status = "warning"
                    page_errors.append("Сбой Gemini при обработке текста")

            elif item["type"] == "image":
                crop_img = item["content"]
                # Передаем эталонный контекст из JSON предыдущей страницы
                vision_description, extracted_table_title, is_table_img = describe_image(
                    crop_img,
                    previous_table_title=last_table_title,
                    previous_page_text=previous_page_context
                )

                if not vision_description.strip():
                    page_status = "warning"
                    page_errors.append("Сбой Gemini при анализе изображения")

                if extracted_table_title:
                    last_table_title = extracted_table_title

                if vision_description.strip():
                    expanded_vision = force_expand_abbreviations(vision_description)
                    page_final_blocks.append(expanded_vision)

        combined_page_text = "\n\n".join(page_final_blocks).strip()
        if not combined_page_text:
            continue

        # 3. Определение метаданных страницы
        page_topic = previous_page_topic or "Не определена"
        page_tags = "Нет тегов"

        metadata = extract_page_metadata(combined_page_text, previous_topic=previous_page_topic)
        if metadata and isinstance(metadata, dict):
            page_topic = metadata.get("topic", page_topic)
            page_tags = metadata.get("tags", page_tags)
        elif last_table_title:
            page_topic = last_table_title
            page_tags = "таблица, кардиология"

        final_json_payload = {
            "page": current_page,
            "analysis_status": page_status,
            "errors": page_errors,
            "topic": page_topic,
            "tags": page_tags,
            "refined_text": combined_page_text
        }

        save_chunk_to_folder(final_json_payload, f"page_{current_page}.json", output_folder)
        print(f"✅ Страница {current_page} успешно сохранена (Тема: '{page_topic}').")