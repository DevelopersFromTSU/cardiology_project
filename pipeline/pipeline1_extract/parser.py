import os
import fitz
import io
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def parse_pdf_pro(pdf_path, start_page=1, end_page=1):
    temp_pdf = "temp_slice.pdf"

    with fitz.open(pdf_path) as src:
        with fitz.open() as dest:
            dest.insert_pdf(src, from_page=start_page - 1, to_page=end_page - 1)
            dest.save(temp_pdf)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = 3.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(temp_pdf)

    doc = fitz.open(temp_pdf)
    page = doc[0]
    page_width = page.rect.width
    page_height = page.rect.height

    matrix_full = fitz.Matrix(3.0, 3.0)
    pix_full = page.get_pixmap(matrix=matrix_full)
    img_data_full = pix_full.tobytes("png")
    full_page_pil_img = Image.open(io.BytesIO(img_data_full))

    y_intervals = []

    # [НОВОЕ]: ЭТАП 1 — Выявляем зоны картинок И таблиц для передачи в Vision
    for item, _ in result.document.iterate_items():
        is_table = getattr(item, "label", "") == "table"
        has_image = hasattr(item, "image") and item.image is not None

        if (has_image or is_table) and hasattr(item, "prov") and item.prov:
            bbox = item.prov[0].bbox
            y_top = page_height - bbox.t
            y_bottom = page_height - bbox.b
            y_intervals.append([min(y_top, y_bottom), max(y_top, y_bottom)])

    # ЭТАП 2 — Склеиваем близкие/пересекающиеся интервалы
    merged_y_intervals = []
    if y_intervals:
        y_intervals.sort(key=lambda x: x[0])
        merged_y_intervals = [y_intervals[0]]

        for current in y_intervals[1:]:
            previous = merged_y_intervals[-1]
            if current[0] <= previous[1] + 40:
                previous[1] = max(previous[1], current[1])
            else:
                merged_y_intervals.append(current)

    raw_elements = []

    # [НОВОЕ]: ЭТАП 3 — Вырезаем картинки (Vision получает абсолютный приоритет на эти зоны)
    for y_min, y_max in merged_y_intervals:
        try:
            crop_rect = fitz.Rect(0, y_min, page_width, y_max)
            matrix = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=matrix, clip=crop_rect)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

            raw_elements.append({
                "type": "image",
                "content": pil_img,
                "full_page_image": full_page_pil_img,
                "y": y_min
            })
        except Exception as e:
            print(f"⚠️ Предупреждение при склеивании картинки: {e}")

    # [НОВОЕ]: ЭТАП 4 — Собираем чистый текст, строго отсекая всё, что попало в зоны картинок/таблиц
    for item, _ in result.document.iterate_items():
        if hasattr(item, "text") and item.text and getattr(item, "label", "") != "table":
            if hasattr(item, "prov") and item.prov:
                bbox = item.prov[0].bbox
                y_pos = page_height - bbox.t

                # Проверяем перекрытие с зонами Vision (допуск +-15 пикселей)
                inside_vision_zone = any((y_min - 15) <= y_pos <= (y_max + 15) for y_min, y_max in merged_y_intervals)

                if not inside_vision_zone:
                    raw_elements.append({
                        "type": "text",
                        "content": item.text,
                        "y": y_pos
                    })

    raw_elements.sort(key=lambda x: x["y"])

    document_elements = []
    for el in raw_elements:
        if el["type"] == "text":
            document_elements.append({"type": "text", "content": el["content"]})
        else:
            document_elements.append(
                {"type": "image", "content": el["content"], "full_page_image": el["full_page_image"]})

    doc.close()
    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)

    return document_elements