import os
import fitz
import io
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def detect_table_zones_by_lines(page, min_line_length=150):
    """
    [НОВОЕ]: Геометрический поиск таблиц по нарисованным линиям в PDF.
    Если на странице есть горизонтальные или вертикальные линии сетки (как на стр. 187),
    функция возвращает их Y-интервалы, даже если Docling их не распознал.
    """
    drawings = page.get_drawings()
    table_y_coords = []

    for draw in drawings:
        rect = draw["rect"]
        # Ищем горизонтальные линии таблицы (ширина > min_line_length, высота тонкая)
        if rect.width > min_line_length and rect.height < 5:
            table_y_coords.append((rect.y0, rect.y1))
        # Ищем длинные вертикальные границы колонок
        elif rect.height > 50 and rect.width < 5:
            table_y_coords.append((rect.y0, rect.y1))

    if not table_y_coords:
        return []

    # Находим общие границы графической сетки на странице
    y_min = min(y[0] for y in table_y_coords)
    y_max = max(y[1] for y in table_y_coords)

    # Если линии занимают существенную высоту (>50 пикселей), считаем это зоной таблицы
    if (y_max - y_min) > 50:
        return [[y_min, y_max]]
    return []


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

    # [ИСПРАВЛЕНО]: ЭТАП 1 — Собираем зоны от Docling
    for item, _ in result.document.iterate_items():
        is_table = getattr(item, "label", "") == "table"
        has_image = hasattr(item, "image") and item.image is not None

        if (has_image or is_table) and hasattr(item, "prov") and item.prov:
            bbox = item.prov[0].bbox
            y_top = page_height - bbox.t
            y_bottom = page_height - bbox.b
            y_intervals.append([min(y_top, y_bottom), max(y_top, y_bottom)])

    # [НОВОЕ]: ЭТАП 1.5 — Добавляем зоны, найденные по физическим линиям сетки PDF
    geometric_table_zones = detect_table_zones_by_lines(page)
    y_intervals.extend(geometric_table_zones)

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

    # ЭТАП 3 — Вырезаем картинки (Vision получает абсолютный приоритет)
    # [НОВОЕ]: ЭТАП 3 — Вырезаем картинки с нижним буфером (+65 px) для гарантированного захвата сносок
    for y_min, y_max in merged_y_intervals:
        try:
            # Расширяем нижнюю границу на 65 пикселей, чтобы захватить подвал со сносками
            buffered_y_max = min(page_height, y_max + 65)
            crop_rect = fitz.Rect(0, y_min, page_width, buffered_y_max)

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

    # ЭТАП 4 — Собираем чистый текст, строго отсекая всё, что попало в зоны таблиц
    for item, _ in result.document.iterate_items():
        if hasattr(item, "text") and item.text and getattr(item, "label", "") != "table":
            if hasattr(item, "prov") and item.prov:
                bbox = item.prov[0].bbox
                y_pos = page_height - bbox.t

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