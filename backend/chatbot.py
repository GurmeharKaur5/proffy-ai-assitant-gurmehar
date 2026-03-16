from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Setup embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

# Use better model
llm = Ollama(
    model="llama3.2:3b",
    temperature=0,
    num_predict=40
)


def ask_question(question):

    docs = db.similarity_search(question, k=2)

    context = ""
    sources = set()

    for doc in docs:
        context += doc.page_content + "\n"
        sources.add(doc.metadata.get("source", "unknown"))

    prompt = f"""
Answer the question ONLY using the context.

Context:
{context}

Question: {question}

Answer in ONE sentence.
"""

    print("\nProffy:", end=" ")

    response = llm.invoke(prompt)

    # Keep only first sentence
    response = response.split(".")[0] + "."

    print(response)
    print(f"\nSources: {', '.join(sources)}")


if __name__ == "__main__":

    print("🎓 Proffy AI Assistant Ready")

    while True:
        query = input("\nYou: ")

        if query.lower() in ["exit", "q"]:
            break

        ask_question(query)