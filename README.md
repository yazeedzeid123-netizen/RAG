# Egyptian AI Learning Assistant (LangChain + Unstructured + Hugging Face)

A bilingual Gradio workspace for asking grounded questions about learning materials. The interface uses a proposed Egyptian learning-platform identity and is not an official Ministry service. Upload PDF, DOCX, or TXT notes, build a local vector index, and ask questions with supporting document excerpts. Embeddings and the Qwen2.5-1.5B-Instruct language model run locally, so model inference does not use a paid API or require an access token.

## Setup
```bash
git clone <your-repo-url> && cd rag-class12
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

The model downloads from Hugging Face the first time generation runs. It needs several GB of disk space and enough memory to run locally; generation speed depends on your computer. Change `HF_LLM_REPO` in `.env` to select another compatible Hugging Face text-generation model.

## Run each step
```bash
python step1_load.py       # Load
python step2_chunk.py      # Chunk
python step3_store.py      # Embed + store
python step4_retrieve.py   # Retrieve
python step5_generate.py   # Generate
python app.py              # Full app
```

## Lesson plan (5 sessions x 40 min, teach 15 min, code 20 min, quiz 5 min)

| Session | Concept (analogy) | Code | Student task |
|---|---|---|---|
| 0. Warm-up | LLMs know a lot but not *your* notes, and sometimes make things up (hallucinate). RAG = open-book exam. | Ask the LLM about a private file and watch it fail | Predict why it fails |
| 1. Load | Photocopying the textbook; metadata = page label and chapter name | `step1_load.py` | Add own PDF; list the element types found |
| 2. Chunk | Cutting notes into flashcards: too big = vague, too small = no meaning | `step2_chunk.py` | Change `max_characters` to 300 and 1500, compare |
| 3. Embed + Store | A library where books sit near similar topics (a map of meaning) | `step3_store.py` | Print vectors for "cat" and "kitten" vs "tax" |
| 4. Retrieve | The librarian fetching the best 3 pages; filters = "only Biology shelf" | `step4_retrieve.py` | Change `k`; try the `source` filter |
| 5. Generate | Open-book exam: question + pages -> answer, with a rule "don't guess" | `step5_generate.py` | Edit the prompt; ask something *not* in the notes |
| 6. Build the app | Putting it all together | `app.py` | Each group indexes their own subject notes |

**Extra ideas:** group project (each team picks a subject), a "hallucination hunt" comparing plain LLM vs RAG answers, quick quiz after each stage.

## Push to GitHub
```bash
git init && git add . && git commit -m "RAG class 12 starter"
git branch -M main
git remote add origin https://github.com/<you>/rag-class12.git
git push -u origin main
```

## Troubleshooting
- Slow first run: sentence-transformers downloads the embedding model (~90 MB) once.
- PDF problems: scanned PDFs need OCR (install tesseract + poppler, use `strategy="hi_res"`).
- Model loading or memory error: check available disk/RAM, or select a smaller compatible model in `.env`.
