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
        return text
    # Удаляем библиографические ссылки
    text = re.sub(r'\[\d+[\d\s,\-]*\]', '', text)
    # Заменяем 3 и более переносов строк на стандартные 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем пробелы и табуляцию в конце строк
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Заменяем 2+ пробела между словами на один
    text = re.sub(r'(?<=\S)[ \t]{2,}', ' ', text)
    return text


def get_smart_chunks(text, chunk_size=1200, chunk_overlap=300):
    headers_to_split_on = [("#", "Header 1")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(md_splits)


def process_and_upload(text_to_upload, page_num, qdrant_client, embedding_model, collection_name):
    cleaned_text = clean_excessive_whitespace(text_to_upload)
    chunks = get_smart_chunks(cleaned_text)

    for chunk in chunks:
        chunk_text = chunk.page_content

        # Получаем плотный и разреженный векторы за один проход модели
        outputs = embedding_model.encode([chunk_text], return_dense=True, return_sparse=True)

        dense_vec = outputs['dense_vecs'][0].tolist()
        sparse_dict = outputs['lexical_weights'][0]

        # Формируем структуру разреженного вектора для Qdrant
        sparse_vec = models.SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()]
        )

        # Записываем точку с именованными векторами
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload={"text": chunk_text, "page": page_num}
        )
        qdrant_client.upsert(collection_name=collection_name, points=[point])


if __name__ == "__main__":
    load_dotenv()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_folder = os.path.join(BASE_DIR, "result")
    collection_name = os.getenv("COLLECTION_NAME", "medical_docs")

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    # Перенастрой инициализацию модели (размерность BGE-M3 = 1024)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Создаем коллекцию с поддержкой разреженных (sparse) и плотных (dense) векторов
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
        for i, filename in enumerate(files):
            filepath = os.path.join(json_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text_to_upload = data.get("refined_text", "")

            page_num = i + 1

            if i < len(files) - 1:
                next_page_num = page_num + 1
                next_filepath = os.path.join(json_folder, files[i + 1])
                with open(next_filepath, 'r', encoding='utf-8') as f_next:
                    next_text = json.load(f_next).get("refined_text", "")
                    raw_overlap = next_text[:500]
                    matches = list(re.finditer(r'[.!?;](?=\s|$)', raw_overlap))

                    if matches:
                        last_punctuation = matches[-1].start()
                        overlap_text = raw_overlap[:last_punctuation + 1]
                    else:
                        last_newline = raw_overlap.rfind('\n')
                        overlap_text = raw_overlap[:last_newline] if last_newline != -1 else raw_overlap

                    text_to_upload += f"\n\n--- НАЧАЛО СТРАНИЦЫ {next_page_num} ---\n\n" + overlap_text

            process_and_upload(
                text_to_upload=text_to_upload,
                page_num=page_num,
                qdrant_client=qdrant,
                embedding_model=model,
                collection_name=collection_name
            )