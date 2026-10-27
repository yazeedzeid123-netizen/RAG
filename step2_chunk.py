"""STEP 2 - CHUNK: split by headings (titles) so each chunk keeps one idea + metadata."""
from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from step1_load import load_elements


def make_chunks(elements_by_file=None):
    elements_by_file = elements_by_file or load_elements()
    docs = []
    for filename, elements in elements_by_file:
        chunks = chunk_by_title(
            elements,
            max_characters=800,
            new_after_n_chars=600,
            combine_text_under_n_chars=150,
        )
        for i, ch in enumerate(chunks):
            meta = ch.metadata.to_dict()
            docs.append(Document(
                page_content=ch.text,
                metadata={                      # Chroma only accepts simple values
                    "source": filename,
                    "page": meta.get("page_number") or 0,
                    "chunk_id": i,
                    "category": ch.category,
                },
            ))
    return docs


if __name__ == "__main__":
    docs = make_chunks()
    print(f"Total chunks: {len(docs)}")
    for d in docs[:3]:
        print("\n---", d.metadata)
        print(d.page_content[:200])
