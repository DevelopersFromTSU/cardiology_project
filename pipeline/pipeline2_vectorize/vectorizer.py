import os
import json
import argparse
import re
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from FlagEmbedding import BGEM3FlagModel


def clean_excessive_whitespace(text):
    if not text:
        return ""
    text = re.sub(r'\[\d+[\d\s,\-]*\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    return text.strip()


# --- НОВАЯ ФУНКЦИЯ АВТОМАТИЧЕСКОЙ СКЛЕЙКИ В ПАМЯТИ ---
def merge_short_pages(page_files_data, min_chars=250):
    """
    Автоматически склеивает короткие страницы (хвосты списков, одиночные фразы)
    с предыдущей полноценной страницей прямо в памяти перед векторизацией.
    """
    if not page_files_data:
        return []

    merged_data = []
    for page_num, text, topic, tags in page_files_data:
        # Если чанк слишком короткий и уже есть куда приклеивать
        if len(text.strip()) < min_chars and merged_data:
            prev_page, prev_text, prev_topic, prev_tags = merged_data[-1]
            print(f"📎 Автосклейка: страница {page_num} ({len(text)} симв.) прикреплена к странице {prev_page}")

            # Дописываем текст к предыдущей странице
            combined_text = f"{prev_text}\n\n{text}"
            merged_data[-1] = (prev_page, combined_text, prev_topic, prev_tags)
        else:
            merged_data.append((page_num, text, topic, tags))

    return merged_data


# -----------------------------------------------------


def get_global_chunks_with_pages(page_files_data, chunk_size=4000, chunk_overlap=400):
    full_text = ""
    page_boundaries = []

    for page_num, raw_text, topic, tags in page_files_data:
        cleaned = clean_excessive_whitespace(raw_text)
        if not cleaned:
            continue

        enriched_text = f"--- ТЕМА СТРАНИЦЫ: {topic} ---\n--- ТЕГИ: {tags} ---\nДАННЫЕ:\n{cleaned}"

        start_idx = len(full_text)
        full_text += enriched_text + "\n\n"
        end_idx = len(full_text)
        page_boundaries.append((start_idx, end_idx, page_num))

    headers_to_split_on = [("#", "Header 1")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(full_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    final_chunks = text_splitter.split_documents(md_splits)

    chunks_with_metadata = []
    for chunk in final_chunks:
        start_char = chunk.metadata.get("start_index", 0)
        assigned_page = page_boundaries[0][2] if page_boundaries else 1
        for p_start, p_end, p_num in page_boundaries:
            if p_start <= start_char < p_end:
                assigned_page = p_num
                break

        chunk.metadata["page"] = assigned_page
        chunks_with_metadata.append(chunk)

    return chunks_with_metadata


def upload_chunks_to_qdrant(chunks, qdrant_client, embedding_model, collection_name, book_name):
    print(f"🔄 Подготовка к загрузке {len(chunks)} чанков в Qdrant...")

    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk.page_content
        page_num = chunk.metadata.get("page", 0)

        outputs = embedding_model.encode([chunk_text], return_dense=True, return_sparse=True)
        dense_vec = outputs['dense_vecs'][0].tolist()
        sparse_dict = outputs['lexical_weights'][0]
        sparse_vec = models.SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()]
        )

        stable_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}_{book_name}_chunk_{i}"))

        point = models.PointStruct(
            id=stable_id,
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload={"text": chunk_text, "page": page_num, "book": book_name}
        )
        qdrant_client.upsert(collection_name=collection_name, points=[point])
        print(f"✅ Загружен чанк {i}/{len(chunks)} (Страница {page_num})")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str, required=True, help="Название папки с книгой")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_folder = os.path.join(BASE_DIR, "result", args.book)
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

    if os.path.exists(json_folder):
        files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])
        raw_page_data = []

        for filename in files:
            filepath = os.path.join(json_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get("refined_text", "")
                topic = data.get("topic", "")
                tags = data.get("tags", "")

            page_match = re.search(r'\d+', filename)
            page_num = int(page_match.group()) if page_match else 1
            raw_page_data.append((page_num, text, topic, tags))

        # 1. Сортируем страницы по порядку
        raw_page_data.sort(key=lambda x: x[0])

        # 2. [НОВАЯ СТРОКА]: Автоматически склеиваем короткие фрагменты в памяти
        page_files_data = merge_short_pages(raw_page_data, min_chars=250)

        # 3. Режем на чанки и векторизуем
        all_chunks = get_global_chunks_with_pages(page_files_data)
        upload_chunks_to_qdrant(all_chunks, qdrant, model, collection_name, args.book)
        print("🎉 Векторизация и загрузка успешно завершены!")