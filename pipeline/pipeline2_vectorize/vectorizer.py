import os
import json
import uuid
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from FlagEmbedding import BGEM3FlagModel


def clean_excessive_whitespace(text):
    if not text:
        return ""
    # Удаляем библиографические ссылки [1, 2]
    text = re.sub(r'\[\d+[\d\s,\-]*\]', '', text)
    # Заменяем 3 и более переносов строк на стандартные 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем пробелы и табуляцию в конце строк
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Заменяем 2+ пробела между словами на один
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    return text.strip()


def get_global_chunks_with_pages(page_files_data, chunk_size=3000, chunk_overlap=300):
    """
    [НОВОЕ]: Объединяет страницы в единый поток, режет на чанки с бесшовным
    перехлестом и точно присваивает номер страницы каждому чанку без дубликатов.
    """
    full_text = ""
    page_boundaries = []  # Хранит кортежи: (start_char, end_char, page_num)

    # 1. Собираем единый текст и карту страниц
    for page_num, raw_text in page_files_data:
        cleaned = clean_excessive_whitespace(raw_text)
        if not cleaned:
            continue

        start_idx = len(full_text)
        full_text += cleaned + "\n\n"
        end_idx = len(full_text)
        page_boundaries.append((start_idx, end_idx, page_num))

    # 2. Сначала учитываем Markdown-заголовки
    headers_to_split_on = [("#", "Header 1")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(full_text)

    # 3. Режем текст с отслеживанием начального индекса (add_start_index=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    final_chunks = text_splitter.split_documents(md_splits)

    # 4. Присваиваем правильную страницу каждому чанку
    chunks_with_metadata = []
    for chunk in final_chunks:
        start_char = chunk.metadata.get("start_index", 0)

        # Находим, в диапазон какой страницы попадает начало чанка
        assigned_page = page_boundaries[0][2] if page_boundaries else 1
        for p_start, p_end, p_num in page_boundaries:
            if p_start <= start_char < p_end:
                assigned_page = p_num
                break

        chunk.metadata["page"] = assigned_page
        chunks_with_metadata.append(chunk)

    return chunks_with_metadata


def upload_chunks_to_qdrant(chunks, qdrant_client, embedding_model, collection_name):
    """
    [НОВОЕ]: Пакетная векторизация и загрузка всех чанков документа.
    """
    print(f"🔄 Подготовка к загрузке {len(chunks)} чанков в Qdrant...")

    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk.page_content
        page_num = chunk.metadata.get("page", 0)

        # Получаем плотный и разреженный векторы за один проход BGE-M3
        outputs = embedding_model.encode([chunk_text], return_dense=True, return_sparse=True)

        dense_vec = outputs['dense_vecs'][0].tolist()
        sparse_dict = outputs['lexical_weights'][0]

        sparse_vec = models.SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()]
        )

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload={"text": chunk_text, "page": page_num}
        )
        qdrant_client.upsert(collection_name=collection_name, points=[point])
        print(f"✅ Загружен чанк {i}/{len(chunks)} (Страница {page_num})")


if __name__ == "__main__":
    load_dotenv()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_folder = os.path.join(BASE_DIR, "result")
    collection_name = os.getenv("COLLECTION_NAME", "medical_docs")

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    print("⏳ Загрузка модели BGE-M3...")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )

    # 1. Читаем все файлы страниц в память и сортируем по номеру страницы
    if os.path.exists(json_folder):
        files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])
        page_files_data = []

        for filename in files:
            filepath = os.path.join(json_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get("refined_text", "")

            page_match = re.search(r'\d+', filename)
            page_num = int(page_match.group()) if page_match else 1
            page_files_data.append((page_num, text))

        # Сортируем строго по возрастанию номеров страниц (page_1, page_2, ..., page_10)
        page_files_data.sort(key=lambda x: x[0])

        # 2. Генерируем чанки со сквозным нахлестом по всему документу
        all_chunks = get_global_chunks_with_pages(page_files_data)

        # 3. Отправляем в базу
        upload_chunks_to_qdrant(all_chunks, qdrant, model, collection_name)
        print("🎉 Векторизация и загрузка успешно завершены!")