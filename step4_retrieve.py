"""STEP 4 - RETRIEVE: find the chunks most similar to the question (+ metadata filter)."""
from step3_store import load_vectorstore


def get_retriever(k=3, source=None):
    kwargs = {"k": k}
    if source:
        kwargs["filter"] = {"source": source}   # metadata filtering
    return load_vectorstore().as_retriever(search_kwargs=kwargs)


if __name__ == "__main__":
    vs = load_vectorstore()
    q = "Where does the Calvin cycle happen?"
    for doc, score in vs.similarity_search_with_score(q, k=3):
        print(f"distance={score:.3f} (lower = closer) | {doc.metadata['source']}")
        print("  ", doc.page_content[:120].replace("\n", " "))
