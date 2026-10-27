"""STEP 3 - EMBED + STORE: turn text into numbers (vectors) and save in Chroma."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import EMBED_MODEL, DB_DIR, COLLECTION
from step2_chunk import make_chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(docs):
    return Chroma.from_documents(
        docs, get_embeddings(),
        collection_name=COLLECTION, persist_directory=DB_DIR,
    )


def load_vectorstore():
    return Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(),
                  persist_directory=DB_DIR)


if __name__ == "__main__":
    vec = get_embeddings().embed_query("photosynthesis")
    print("One word becomes", len(vec), "numbers. First 5:", [round(x, 3) for x in vec[:5]])
    vs = build_vectorstore(make_chunks())
    print("Stored chunks in Chroma:", vs._collection.count())
