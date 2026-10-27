"""STEP 6 - APP: a simple web UI with Gradio. Run: python app.py"""
import shutil
from pathlib import Path
import gradio as gr
from config import DATA_DIR
from step2_chunk import make_chunks
from step3_store import build_vectorstore, load_vectorstore
from step5_generate import build_chain, get_retriever

state = {"chain": None, "retriever": None}


def index_files(files):
    Path(DATA_DIR).mkdir(exist_ok=True)
    for f in files or []:
        shutil.copy(f.name if hasattr(f, "name") else f, DATA_DIR)
    vs = build_vectorstore(make_chunks())
    return f"Indexed {vs._collection.count()} chunks. Now ask questions!"


def ask(question, source_filter):
    retriever = get_retriever(k=3, source=source_filter or None)
    chain = build_chain(retriever)
    answer = chain.invoke(question)
    sources = "\n".join(f"- {d.metadata['source']} (page {d.metadata['page']}, chunk {d.metadata['chunk_id']})"
                        for d in retriever.invoke(question))
    return answer, sources


with gr.Blocks(title="Class 12 RAG Tutor") as demo:
    gr.Markdown("# Class 12 RAG Tutor\nUpload notes (PDF/DOCX/TXT), index them, then ask questions.")
    files = gr.File(file_count="multiple", label="Upload notes")
    status = gr.Textbox(label="Status", interactive=False)
    gr.Button("1. Build index").click(index_files, files, status)
    q = gr.Textbox(label="2. Your question")
    src = gr.Textbox(label="Optional: only search this file (e.g. sample_notes.txt)")
    ans = gr.Textbox(label="Answer", lines=6)
    cites = gr.Textbox(label="Sources used (metadata)", lines=4)
    gr.Button("Ask").click(ask, [q, src], [ans, cites])

if __name__ == "__main__":
    demo.launch()
