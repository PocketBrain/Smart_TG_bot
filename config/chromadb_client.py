from loguru import logger
import pandas as pd
import chromadb
from PyPDF2 import PdfReader
import re
import os
from docx import Document
from config.ml import sentence_model
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        # Инициализация списка частей текста
        chunks = []
        current_chunk = ""
        # Разделение текста на предложения с помощью регулярного выражения
        sentences = re.split(r'(?<=[^A-ZА-Я]\.)\s+(?=[A-ZА-Я])', text)

        for sentence in sentences:
            # Проверка, поместится ли предложение в текущую часть
            if len(current_chunk) + len(sentence.split()) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = " ".join(sentence.split()[-self.chunk_overlap:]) + " "

        # Добавление последней части, если она есть
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


def read_text_from_file(file_path):
    if file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            return '\n'.join(reader.pages[i].extract_text() for i in range(len(reader.pages)))
    else:
        raise ValueError("Unsupported file format. Only .docx and .pdf are supported.")

def load_documents(file_paths):
    documents = []

    for file_path in file_paths:
        print(file_path)
        file_contents = read_text_from_file(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(file_contents)
        documents.extend([{'text': chunk, 'file_path': file_path, 'page_number': i} for i, chunk in enumerate(chunks)])

    return documents

def get_file_paths(directory):
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_extension = os.path.splitext(file_path)

            # Если расширение файла не .pdf, изменить имя файла и добавить .pdf
            if file_extension.lower() != '.pdf':
                new_file_path = file_name + '.pdf'
                os.rename(file_path, new_file_path)
                file_paths.append(new_file_path)
            else:
                file_paths.append(file_path)

    return file_paths

def collectionUpsert(df):
  texts = df["text"].tolist()
  file_path = df["file_path"].tolist()
  text_embeddings = df["text_embeddings"].tolist()
  ids = [str(i) for i in range(1, len(df['text'])+1)]
  collection.upsert(
      ids=ids,
      embeddings=text_embeddings,
      metadatas=[{"source": "test", "file_path": txt} for txt in file_path],
      documents=texts
  )

  return collection

def calculate_embedding(txt, model):
    embedding = model.encode(txt, normalize_embeddings=True)
    return embedding.tolist()

def fill_documents(documents):
    df = pd.DataFrame()
    df["text"] = [doc['text'] for doc in documents]
    df["file_path"] = [doc['file_path'] for doc in documents]
    df["page_number"] = [doc['page_number'] for doc in documents]
    df["text_embeddings"] = df["text"].apply(calculate_embedding, args=(sentence_model,))
    return df



logger.info("initializing the chromadb")
client = chromadb.PersistentClient(path="chroma/")
collection = client.get_or_create_collection("test")

directory_path = 'datasets/'
#file_paths = get_file_paths(directory_path)
documents = load_documents(['datasets/test.pdf'])


df = fill_documents(documents)
collection_result = collectionUpsert(df)