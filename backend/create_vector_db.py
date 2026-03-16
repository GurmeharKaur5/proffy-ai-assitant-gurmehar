import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = []

data_path = "data"

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            course_name = root.split("/")[-1]

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "course": course_name.lower(),
                        "source": path
                    }
                )
            )

print("Documents loaded:", len(documents))


# SPLIT DOCUMENTS INTO CHUNKS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# EMBEDDINGS
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# CREATE VECTOR DB
db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="vector_db"
)

db.persist()

print("Vector database created successfully!")