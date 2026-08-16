import streamlit as st
import os
import json
import re
import subprocess
import sys
from pathlib import Path

# 1. Настройка путей относительно проекта
BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"
RESULT_DIR = PIPELINE_DIR / "result"
DATA_DIR = PIPELINE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="CardiologyV2 Admin", layout="wide")
st.title("🫀 CardiologyV2: Панель управления RAG-данными")


def extract_page_number(filename):
    match = re.search(r'page_(\d+)\.json', filename)
    return int(match.group(1)) if match else 0


# --- Пользовательские стили ---
st.markdown(
    """
    <style>
    div.stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        padding: 15px !important;
        font-weight: 400 !important;
    }
    div.stTextInput input {
        font-size: 15px !important;
        font-weight: 500 !important;
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

    if "current_page_index" not in st.session_state:
        st.session_state.current_page_index = 0
    if "selected_book" not in st.session_state:
        st.session_state.selected_book = None

    if books:
        selected_book = st.selectbox("📚 Выберите обработанную книгу", books)

        if st.session_state.selected_book != selected_book:
            st.session_state.selected_book = selected_book
            st.session_state.current_page_index = 0

        book_path = RESULT_DIR / selected_book

        json_files = sorted(
            [f.name for f in book_path.iterdir() if f.name.startswith('page_') and f.name.endswith('.json')],
            key=extract_page_number
        )

        if json_files:
            if st.session_state.current_page_index >= len(json_files):
                st.session_state.current_page_index = len(json_files) - 1

            def format_page_name(filename):
                page_num = extract_page_number(filename)
                try:
                    with open(book_path / filename, "r", encoding="utf-8") as temp_f:
                        temp_data = json.load(temp_f)
                        if temp_data.get("is_verified", False):
                            return f"✅ Страница {page_num}"
                        if temp_data.get("analysis_status") == "warning":
                            errors = ", ".join(temp_data.get("errors", []))
                            return f"⚠️ Страница {page_num} ({errors})"
                except Exception:
                    pass
                return f"📄 Страница {page_num}"

            # Навигация по страницам
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

            with nav_col1:
                st.write("")
                st.write("")
                if st.button("⬅️ Предыдущая", use_container_width=True):
                    if st.session_state.current_page_index > 0:
                        st.session_state.current_page_index -= 1
                        st.rerun()

            with nav_col2:
                selected_file = st.selectbox(
                    "Выберите страницу",
                    options=json_files,
                    index=st.session_state.current_page_index,
                    format_func=format_page_name
                )

            with nav_col3:
                st.write("")
                st.write("")
                if st.button("Следующая ➡️", type="primary", use_container_width=True):
                    if st.session_state.current_page_index < len(json_files) - 1:
                        st.session_state.current_page_index += 1
                        st.rerun()

            current_selected_index = json_files.index(selected_file)
            if current_selected_index != st.session_state.current_page_index:
                st.session_state.current_page_index = current_selected_index
                st.rerun()

            file_path = book_path / selected_file

            with open(file_path, "r", encoding="utf-8") as f:
                page_data = json.load(f)

                # Индикатор статуса парсинга
                status = page_data.get("analysis_status", "success")
                errors = page_data.get("errors", [])

                if status == "success" and not errors:
                    st.success("🟢 **Статус обработки:** Успешно — данные и таблицы распознаны без сбоев.")
                elif status == "warning" or errors:
                    error_text = ", ".join(errors) if errors else "Обнаружены предупреждения при парсинге"
                    st.warning(f"⚠️ **Статус обработки:** Требует внимания | **Ошибки:** {error_text}")
                else:
                    st.error(f"🔴 **Статус обработки:** Критический сбой ({status})")

                col_text, col_img = st.columns(2)

                with col_text:
                    st.subheader("📝 Редактирование данных")
                    editor_tab, preview_tab = st.tabs(["✏️ Редактор", "👁️ Превью Markdown"])

                    with editor_tab:
                        with st.form(key=f"edit_form_{selected_book}_{selected_file}"):
                            new_topic = st.text_input("Тема страницы (topic)", page_data.get("topic", ""))
                            new_tags = st.text_input("Теги (tags)", page_data.get("tags", ""))
                            new_text = st.text_area(
                                "Очищенный текст (refined_text)",
                                page_data.get("refined_text", ""),
                                height=600
                            )
                            is_verified = st.checkbox(
                                "✅ Подтверждаю, что страница проверена и готова для базы",
                                value=page_data.get("is_verified", False)
                            )
                            submit_button = st.form_submit_button(
                                "💾 Сохранить изменения в JSON",
                                use_container_width=True
                            )

                            if submit_button:
                                page_data["topic"] = new_topic
                                page_data["tags"] = new_tags
                                page_data["refined_text"] = new_text
                                page_data["is_verified"] = is_verified
                                with open(file_path, "w", encoding="utf-8") as save_f:
                                    json.dump(page_data, save_f, ensure_ascii=False, indent=4)
                                st.success("Файл успешно обновлен!")
                                st.rerun()

                    with preview_tab:
                        st.markdown(f"### Тема: {page_data.get('topic', 'Без темы')}")
                        st.caption(f"**Теги:** {page_data.get('tags', 'Нет тегов')}")
                        st.divider()
                        st.markdown(page_data.get("refined_text", ""))

                with col_img:
                    st.subheader("🖼️ Оригинал страницы")
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
                            st.image(img, use_container_width=True)
                            doc.close()
                        except Exception as e:
                            st.error(f"Не удалось загрузить превью PDF: {e}")
                    else:
                        st.warning(f"Исходный файл '{selected_book}.pdf' не найден в папке data.")

# --- Вкладка 2: Загрузка и парсинг ---
with tab_upload:
    st.header("Управление книгами (Папка data)")

    uploaded_file = st.file_uploader("📥 Если нужной книги нет, загрузите PDF сюда:", type=["pdf"])
    if uploaded_file:
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Файл {uploaded_file.name} успешно сохранен в папку data!")

    st.divider()

    pdf_files = [f.name for f in DATA_DIR.iterdir() if f.name.endswith('.pdf')]

    if pdf_files:
        selected_pdf = st.selectbox("📖 Выберите книгу из папки data для парсинга", pdf_files)

        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Начальная страница", min_value=1, value=1)
        with col2:
            end_page = st.number_input("Конечная страница", min_value=1, value=5)

        if "current_parsing_page" not in st.session_state:
            st.session_state.current_parsing_page = None

        if st.session_state.current_parsing_page is None:
            if st.button("▶️ Запустить парсинг (main.py)", use_container_width=True):
                st.session_state.current_parsing_page = start_page
                st.rerun()
        else:
            current = st.session_state.current_parsing_page

            progress_placeholder = st.empty()
            stop_btn_placeholder = st.empty()

            progress = (current - start_page) / (end_page - start_page + 1) if end_page >= start_page else 1.0
            progress_placeholder.progress(progress, text=f"⏳ Идет обработка страницы {current} из {end_page}...")

            if stop_btn_placeholder.button("🛑 Остановить парсинг", type="primary", use_container_width=True):
                st.session_state.current_parsing_page = None
                st.error("Парсинг был принудительно остановлен пользователем.")
                st.rerun()

            st.markdown("### 🖥️ Живой лог обработки:")
            log_container = st.empty()

            class StreamlitConsole:
                def __init__(self, placeholder):
                    self.placeholder = placeholder
                    self.buffer = []

                def write(self, text):
                    if text.strip():
                        self.buffer.append(text.strip())
                        display_text = "\n".join(self.buffer[-20:])
                        self.placeholder.code(display_text, language="bash")

                def flush(self):
                    pass

            old_stdout = sys.stdout
            sys.stdout = StreamlitConsole(log_container)

            try:
                if str(BASE_DIR) not in sys.path:
                    sys.path.append(str(BASE_DIR))
                from pipeline.pipeline1_extract.main import run_pipeline

                book_path = DATA_DIR / selected_pdf
                book_name = selected_pdf.replace(".pdf", "")
                output_folder = RESULT_DIR / book_name

                run_pipeline(
                    book_path=str(book_path),
                    output_folder=str(output_folder),
                    start_page=current,
                    end_page=current
                )
            except Exception as e:
                print(f"❌ Возникла критическая ошибка: {e}")
            finally:
                sys.stdout = old_stdout

            if current < end_page:
                st.session_state.current_parsing_page += 1
                st.rerun()
            else:
                st.session_state.current_parsing_page = None
                progress_placeholder.empty()
                stop_btn_placeholder.empty()
                st.success(f"✅ Парсинг успешно завершен! Результаты в папке {book_name}")
                if st.button("Ок, скрыть логи"):
                    st.rerun()

    else:
        st.info("Папка data пуста. Пожалуйста, загрузите PDF-файл.")

# --- Вкладка 3: Экспорт в Qdrant ---
with tab_export:
    st.header("Векторизация и загрузка в Qdrant")

    books_for_export = [d.name for d in RESULT_DIR.iterdir() if d.is_dir()]

    if books_for_export:
        selected_export_book = st.selectbox("📚 Выберите книгу для векторизации", books_for_export)

        if st.button("🚀 Запустить vectorizer.py для выбранной книги"):
            with st.spinner(f"Создание эмбеддингов для '{selected_export_book}' и отправка в БД..."):
                vectorizer_script = PIPELINE_DIR / "pipeline2_vectorize" / "vectorizer.py"
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