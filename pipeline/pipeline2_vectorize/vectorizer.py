import os
import json
import argparse
import re
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel


def clean_excessive_whitespace(text: str) -> str:
    if not text:
        return ""
    # Удалена регулярка, убивающая [1]
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    return text.strip()


def get_global_chunks_with_pages(page_files_data: list, book_name: str, chunk_size: int = 1500, chunk_overlap: int = 250) -> list:
    """
    Собирает все страницы в единый поток книги, режет со сквозным нахлестом через границы страниц
    и точно определяет все страницы, затронутые каждым чанком.
    """
    full_text = ""
    page_spans = []

    # 1. Формируем непрерывную книгу и точную карту смещений символов
    for page_num, raw_text, topic, tags in page_files_data:
        cleaned = clean_excessive_whitespace(raw_text)
        if not cleaned:
            continue

        start_idx = len(full_text)
        full_text += cleaned + "\n\n"
        end_idx = len(full_text)

        page_spans.append({
            "start": start_idx,
            "end": end_idx,
            "page": page_num,
            "topic": topic,
            "tags": tags
        })

    if not full_text:
        return []

    # 2. Единый сплиттер со сквозным нахлестом и сохранением заголовков Markdown
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " "]
    )

    raw_chunks = text_splitter.create_documents([full_text])
    final_chunks = []

    # 3. Сопоставляем каждый чанк со всеми пересекаемыми страницами
    for chunk in raw_chunks:
        chunk_start = chunk.metadata.get("start_index", 0)
        chunk_end = chunk_start + len(chunk.page_content)

        overlapping_pages = []
        chunk_topics = set()
        chunk_tags = set()

        for span in page_spans:
            # Проверка пересечения отрезка чанка [chunk_start, chunk_end] с отрезком страницы [span.start, span.end]
            if max(chunk_start, span["start"]) < min(chunk_end, span["end"]):
                overlapping_pages.append(span["page"])
                if span["topic"] and span["topic"] not in ["Не определена", "Медицинские данные"]:
                    chunk_topics.add(span["topic"])
                if span["tags"] and span["tags"] != "Нет тегов":
                    chunk_tags.add(span["tags"])

        primary_page = overlapping_pages[0] if overlapping_pages else 1
        pages_str = ", ".join(map(str, overlapping_pages)) if overlapping_pages else str(primary_page)
        topic_str = "; ".join(chunk_topics) if chunk_topics else "Кардиология"
        tags_str = ", ".join(chunk_tags) if chunk_tags else "рекомендации"

        # 4. Вшиваем контекстный префикс в тело текста чанка для эмбеддера
        context_prefix = f"Трактат: {book_name} | Стр: {pages_str} | Тема: {topic_str}"
        chunk.page_content = f"{context_prefix}\n\n{chunk.page_content}"

        chunk.metadata = {
            "page": primary_page,
            "pages": overlapping_pages,
            "book": book_name,
            "topic": topic_str,
            "tags": tags_str
        }
        final_chunks.append(chunk)

    return final_chunks


def upload_chunks_to_qdrant(chunks: list, qdrant_client: QdrantClient, embedding_model: BGEM3FlagModel, collection_name: str, book_name: str):
    print(f"🔄 Подготовка к загрузке {len(chunks)} сквозных чанков в Qdrant...")

    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk.page_content
        page_num = chunk.metadata.get("page", 0)
        pages_list = chunk.metadata.get("pages", [page_num])
        topic = chunk.metadata.get("topic", "")
        tags = chunk.metadata.get("tags", "")

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
            payload={
                "text": chunk_text,
                "page": page_num,
                "pages": pages_list,
                "book": book_name,
                "topic": topic,
                "tags": tags
            }
        )
        qdrant_client.upsert(collection_name=collection_name, points=[point])
        print(f"✅ Загружен чанк {i}/{len(chunks)} (Стр. {pages_list})")


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

        # Сортируем страницы по порядку
        raw_page_data.sort(key=lambda x: x[0])

        # Сквозное разбиение с нахлестом через границы страниц
        all_chunks = get_global_chunks_with_pages(raw_page_data, args.book, chunk_size=1500, chunk_overlap=250)
        upload_chunks_to_qdrant(all_chunks, qdrant, model, collection_name, args.book)
        print("🎉 Сквозная векторизация книги завершена!")