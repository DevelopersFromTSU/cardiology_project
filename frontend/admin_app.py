import streamlit as st
import os
import json
import re
import subprocess
import sys
from pathlib import Path

# 1. Строим правильные пути с учетом папки pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"
RESULT_DIR = PIPELINE_DIR / "result"
DATA_DIR = PIPELINE_DIR / "data"

# Принудительно создаем папки, если их случайно удалили
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="CardiologyV2 Admin", layout="wide")
st.title("🫀 CardiologyV2: Панель управления RAG-данными")


def extract_page_number(filename):
    match = re.search(r'page_(\d+)\.json', filename)
    return int(match.group(1)) if match else 0


# --- Пользовательские CSS стили для улучшения читабельности ---
st.markdown(
    """
    <style>
    /* Настраиваем многострочное поле (Очищенный текст) */
    div.stTextArea textarea {
        font-size: 16px !important;       /* Увеличиваем базовый размер шрифта */
        line-height: 1.6 !important;      /* Добавляем "воздух" между строками */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; /* Ставим мягкий системный шрифт */
        padding: 15px !important;         /* Добавляем внутренние отступы от краев рамки */
        font-weight: 400 !important;      /* Делаем начертание чуть более четким */
    }

    /* Настраиваем однострочные поля (Тема и Теги) */
    div.stTextInput input {
        font-size: 15px !important;       /* Чуть увеличиваем заголовки для гармонии */
        font-weight: 500 !important;      /* Делаем их слегка полужирными */
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab_edit, tab_upload, tab_export = st.tabs([
    "1. Редактирование страниц",
    "2. Управление книгами (Парсинг)",
    "3. Экспорт в Qdrant"
])

# --- Вкладка 1: Просмотр и редактирование JSON ---
with tab_edit:
    st.header("Редактирование обработанных страниц")

    books = [d.name for d in RESULT_DIR.iterdir() if d.is_dir()]

    if books:
        selected_book = st.selectbox("📚 Выберите обработанную книгу", books)
        book_path = RESULT_DIR / selected_book

        json_files = sorted(
            [f.name for f in book_path.iterdir() if f.name.startswith('page_') and f.name.endswith('.json')],
            key=extract_page_number
        )

        if json_files:
            selected_file = st.selectbox(
                "📄 Выберите страницу",
                options=json_files,
                format_func=lambda x: f"Страница {extract_page_number(x)}"
            )
            file_path = book_path / selected_file

            with open(file_path, "r", encoding="utf-8") as f:
                page_data = json.load(f)

                # Разделяем экран на две СТРОГО равные колонки (50% и 50% ширины)
                col_text, col_img = st.columns(2)

                # ЛЕВАЯ КОЛОНКА: Редактор JSON (col_text идет первой)
                with col_text:
                    st.subheader("📝 Редактирование данных")

                    # Оберните поля ввода в st.form с уникальным ключом
                    with st.form(key=f"edit_form_{selected_book}_{selected_file}"):
                        new_topic = st.text_input("Тема страницы (topic)", page_data.get("topic", ""))
                        new_tags = st.text_input("Теги (tags)", page_data.get("tags", ""))

                        # Текстовое поле больше не будет триггерить перезагрузку при потере фокуса
                        new_text = st.text_area("Очищенный текст (refined_text)", page_data.get("refined_text", ""),
                                                height=800)

                        # Внутри формы обязательно нужно использовать st.form_submit_button вместо st.button
                        submit_button = st.form_submit_button("💾 Сохранить изменения в JSON", use_container_width=True)

                        if submit_button:
                            page_data["topic"] = new_topic
                            page_data["tags"] = new_tags
                            page_data["refined_text"] = new_text

                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(page_data, f, ensure_ascii=False, indent=4)
                            st.success("Файл успешно обновлен!")

                # ПРАВАЯ КОЛОНКА: Скриншот оригинала (col_img идет второй)
                with col_img:
                    st.subheader("👁️ Оригинал страницы")

                    pdf_path = DATA_DIR / f"{selected_book}.pdf"
                    page_num = extract_page_number(selected_file)

                    if pdf_path.exists() and page_num > 0:
                        import fitz
                        import io
                        from PIL import Image

                        try:
                            doc = fitz.open(pdf_path)
                            page = doc[page_num - 1]

                            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                            img = Image.open(io.BytesIO(pix.tobytes("png")))

                            # Картинка автоматически займет 100% ширины правой колонки
                            st.image(img, use_container_width=True)
                            doc.close()
                        except Exception as e:
                            st.error(f"Не удалось загрузить превью PDF: {e}")
                    else:
                        st.warning(f"Исходный файл '{selected_book}.pdf' не найден в папке data.")

# --- Вкладка 2: Загрузка и парсинг ---
with tab_upload:
    st.header("Управление книгами (Папка data)")

    # Блок загрузки нового PDF
    uploaded_file = st.file_uploader("📥 Если нужной книги нет, загрузите PDF сюда:", type=["pdf"])
    if uploaded_file:
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Файл {uploaded_file.name} успешно сохранен в папку data!")

    st.divider()

    # Блок выбора книги и парсинга
    pdf_files = [f.name for f in DATA_DIR.iterdir() if f.name.endswith('.pdf')]

    if pdf_files:
        selected_pdf = st.selectbox("📖 Выберите книгу из папки data для парсинга", pdf_files)

        # Настройка диапазона страниц
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Начальная страница", min_value=1, value=1)
        with col2:
            end_page = st.number_input("Конечная страница", min_value=1, value=5)

        if st.button("▶️ Запустить парсинг (main.py)"):
            with st.spinner(f"Идет обработка страниц {start_page}-{end_page}. Пожалуйста, подождите..."):
                # Добавляем корень проекта в системный путь, чтобы импорты в main.py не сломались
                if str(BASE_DIR) not in sys.path:
                    sys.path.append(str(BASE_DIR))

                # Импорт именно из нужной папки, опираясь на скриншот
                from pipeline.pipeline1_extract.main import run_pipeline

                book_path = DATA_DIR / selected_pdf
                book_name = selected_pdf.replace(".pdf", "")
                output_folder = RESULT_DIR / book_name

                run_pipeline(
                    book_path=str(book_path),
                    output_folder=str(output_folder),
                    start_page=int(start_page),
                    end_page=int(end_page)
                )
                st.success(f"Парсинг завершен! Результаты сохранены в папку {book_name}")
    else:
        st.info("Папка data пуста. Пожалуйста, загрузите PDF-файл.")

# --- Вкладка 3: Экспорт в Qdrant ---
with tab_export:
    st.header("Векторизация и загрузка в Qdrant")

    # Ищем папки с обработанными книгами
    books_for_export = [d.name for d in RESULT_DIR.iterdir() if d.is_dir()]

    if books_for_export:
        selected_export_book = st.selectbox("📚 Выберите книгу для векторизации", books_for_export)

        if st.button("🚀 Запустить vectorizer.py для выбранной книги"):
            with st.spinner(f"Создание эмбеддингов для '{selected_export_book}' и отправка в БД..."):

                vectorizer_script = PIPELINE_DIR / "pipeline2_vectorize" / "vectorizer.py"

                # Добавляем аргумент --book при вызове скрипта
                result = subprocess.run(
                    ["python", str(vectorizer_script), "--book", selected_export_book],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    st.success(f"Данные книги '{selected_export_book}' успешно загружены в базу!")
                    st.code(result.stdout)
                else:
                    st.error("Произошла ошибка при загрузке:")
                    st.code(result.stderr)
    else:
        st.info("Нет обработанных книг в папке result для экспорта.")