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

    # Сюда будем складывать ВСЕ элементы с их координатой Y
    raw_elements = []

    # Сюда складываем координаты картинок для склеивания
    y_intervals = []

    # 1. ПЕРВЫЙ ПРОХОД: Собираем текст и рамки картинок с привязкой к высоте
    for item, _ in result.document.iterate_items():
        if hasattr(item, "text") and item.text:
            # Находим Y-координату текстового блока
            y_pos = 0
            if hasattr(item, "prov") and item.prov:
                bbox = item.prov[0].bbox
                # Вычисляем расстояние от верхнего края страницы (меньше цифра = выше на листе)
                y_pos = page_height - bbox.t

            raw_elements.append({
                "type": "text",
                "content": item.text,
                "y": y_pos
            })

        elif hasattr(item, "image") and item.image is not None and hasattr(item, "prov") and item.prov:
            bbox = item.prov[0].bbox
            y_top = page_height - bbox.t
            y_bottom = page_height - bbox.b
            y_min = min(y_top, y_bottom)
            y_max = max(y_top, y_bottom)
            y_intervals.append([y_min, y_max])

    # 2. МАТЕМАТИКА: Склеиваем пересекающиеся картинки (чтобы таблица и легенда были вместе)
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

    # 3. ВТОРОЙ ПРОХОД: Вырезаем финальные картинки и добавляем их в общий котел с Y-координатой
    for y_min, y_max in merged_y_intervals:
        try:
            x0 = 0
            x1 = page_width
            crop_rect = fitz.Rect(x0, y_min, x1, y_max)

            matrix = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=matrix, clip=crop_rect)
            img_data = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_data))

            raw_elements.append({
                "type": "image",
                "content": pil_img,
                "full_page_image": full_page_pil_img,
                "y": y_min  # Записываем верхнюю границу склеенной картинки
            })
        except Exception as e:
            print(f"⚠️ Предупреждение при склеивании картинки: {e}")

    # 4. ВОССТАНОВЛЕНИЕ ПОРЯДКА: Сортируем все элементы (текст и картинки) сверху вниз
    raw_elements.sort(key=lambda x: x["y"])

    # Очищаем временную координату 'y' перед отправкой в main.py
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