"""Bilingual Egyptian AI Learning Assistant interface for the local RAG pipeline."""
import filecmp
import html
import shutil
from pathlib import Path

import gradio as gr

from config import DATA_DIR, DB_DIR
from step1_load import load_elements
from step2_chunk import make_chunks
from step3_store import build_vectorstore, load_vectorstore
from step5_generate import build_chain, get_retriever


SUPPORTED_TYPES = {".pdf", ".docx", ".txt"}


def _stored_files():
    folder = Path(DATA_DIR)
    if not folder.exists():
        return []
    return sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() in SUPPORTED_TYPES),
        key=lambda path: path.name.casefold(),
    )


def _library_html():
    files = _stored_files()
    if not files:
        return (
            '<div class="empty-state"><span class="empty-icon">▤</span>'
            '<strong>Your library is ready</strong>'
            '<span>Upload PDF, DOCX or TXT learning materials to begin.</span></div>'
        )

    rows = []
    for path in files:
        size = path.stat().st_size
        if size < 1024:
            size_label = f"{size} B"
        else:
            size_label = f"{size / 1024:.1f} KB"
        suffix = html.escape(path.suffix[1:].upper())
        name = html.escape(path.name)
        rows.append(
            '<div class="library-row">'
            f'<span class="file-type">{suffix}</span>'
            f'<span class="file-name" title="{name}">{name}</span>'
            f'<span class="file-size">{size_label}</span></div>'
        )
    return '<div class="library-list">' + "".join(rows) + "</div>"


def _stats_html(chunk_count=None):
    files = len(_stored_files())
    chunks = "—" if chunk_count is None else str(chunk_count)
    status = "Index count updates after indexing" if chunk_count is None else "Vector index ready"
    return (
        '<div class="stats-grid">'
        f'<div class="stat-card"><span class="stat-value">{files}</span>'
        '<span class="stat-label">Documents</span></div>'
        f'<div class="stat-card"><span class="stat-value">{chunks}</span>'
        '<span class="stat-label">Indexed chunks</span></div>'
        f'<div class="stat-card stat-status"><span class="status-dot"></span>'
        f'<span>{html.escape(status)}</span></div></div>'
    )


STAGES = [
    ("Document upload", "رفع المستندات"),
    ("Text extraction & chunking", "استخراج النصوص وتقسيمها"),
    ("Embeddings & vector storage", "الفهرسة وتخزين المتجهات"),
    ("Intelligent retrieval & answers", "البحث الذكي والإجابات"),
    ("Sources & answer verification", "المصادر والتحقق من الإجابات"),
]


def _stepper_html(current=1, completed=()):
    completed = set(completed)
    cards = []
    for number, (english, arabic) in enumerate(STAGES, start=1):
        state = "complete" if number in completed else "current" if number == current else "pending"
        icon = "✓" if state == "complete" else f"{number:02d}"
        cards.append(
            f'<div class="step {state}">'
            f'<span class="step-number">{icon}</span>'
            '<span class="step-copy">'
            f'<span class="step-en">{english}</span>'
            f'<span class="step-ar" lang="ar" dir="rtl">{arabic}</span>'
            '</span></div>'
        )
    return '<div class="stepper">' + "".join(cards) + "</div>"


def _upload_summary(files):
    if not files:
        return '<span class="upload-note">Choose one or more PDF, DOCX or TXT files.</span>'

    entries = []
    invalid = []
    for item in files:
        source = Path(getattr(item, "name", item))
        if source.suffix.lower() not in SUPPORTED_TYPES:
            invalid.append(source.name)
            continue
        size = source.stat().st_size if source.is_file() else 0
        label = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
        entries.append(
            f'<span class="upload-file">{html.escape(source.name)}'
            f'<small>{label}</small></span>'
        )
    if invalid:
        entries.append(
            '<span class="validation-error">Unsupported file type: '
            + ", ".join(html.escape(name) for name in invalid)
            + "</span>"
        )
    if not entries:
        return '<span class="validation-error">Select PDF, DOCX or TXT files only.</span>'
    return '<div class="upload-summary">' + "".join(entries) + "</div>"


def _unique_destination(folder, source):
    """Choose a stable name without replacing a different document."""
    candidate = folder / source.name
    version = 1
    while candidate.exists():
        if filecmp.cmp(source, candidate, shallow=False):
            return candidate, False
        candidate = folder / f"{source.stem} ({version}){source.suffix}"
        version += 1
    return candidate, True


def _vector_count(store=None):
    if store is not None:
        return store._collection.count()
    if (Path(DB_DIR) / "chroma.sqlite3").is_file():
        return load_vectorstore()._collection.count()
    return 0


def index_files(files):
    if not files:
        yield "Select at least one document before building the knowledge base.", _library_html(), _stats_html(), gr.skip()
        return

    folder = Path(DATA_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    errors = []
    for item in files:
        source = Path(getattr(item, "name", item))
        if source.suffix.lower() not in SUPPORTED_TYPES:
            errors.append(f"{source.name}: unsupported file type")
            continue
        if not source.is_file():
            errors.append(f"{source.name}: file is unavailable")
            continue
        destination, should_copy = _unique_destination(folder, source)
        if should_copy and source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        names.append(destination.name)

    if errors:
        message = "Some files were not accepted: " + "; ".join(errors)
        if not names:
            yield message, _library_html(), _stats_html(), gr.skip()
            return
    else:
        message = "Reading documents and preparing the knowledge base…"
    yield message, _library_html(), _stats_html(), _stepper_html(current=2, completed=(1,))

    store = load_vectorstore() if (Path(DB_DIR) / "chroma.sqlite3").is_file() else None
    indexed_names = set()
    if store is not None:
        for name in set(names):
            matches = store._collection.get(where={"source": name}, include=["metadatas"])
            if matches.get("ids"):
                indexed_names.add(name)

    new_names = set(names) - indexed_names
    if new_names:
        elements = [
            (name, parsed)
            for name, parsed in load_elements(str(folder))
            if name in new_names
        ]
        chunks = make_chunks(elements)
    else:
        chunks = []

    if new_names and not chunks:
        yield (
            "No text chunks were extracted. Check that the selected documents contain readable text.",
            _library_html(),
            _stats_html(_vector_count(store)),
            _stepper_html(current=2, completed=(1,)),
        )
        return

    if chunks:
        yield (
            f"Extracted {len(chunks)} chunks. Writing embeddings to the vector store…",
            _library_html(),
            _stats_html(),
            _stepper_html(current=3, completed=(1, 2)),
        )
        store = build_vectorstore(chunks)

    total = _vector_count(store)
    unchanged = len(names) - len(new_names)
    suffix = f" {unchanged} already-indexed document(s) were left unchanged." if unchanged else ""
    if errors:
        suffix += " Skipped: " + "; ".join(errors) + "."
    yield (
        f"Knowledge base ready: {total} indexed chunks.{suffix}",
        _library_html(),
        _stats_html(total),
        _stepper_html(current=4, completed=(1, 2, 3)),
    )


def remove_document(filename):
    if not filename:
        return "Choose a document in the library first.", _library_html(), _stats_html(), _stepper_html()

    target = (Path(DATA_DIR) / Path(filename).name).resolve()
    data_folder = Path(DATA_DIR).resolve()
    if target.parent != data_folder or not target.is_file() or target.suffix.lower() not in SUPPORTED_TYPES:
        return "The selected document is not available in the library.", _library_html(), _stats_html(), _stepper_html()

    count = None
    if (Path(DB_DIR) / "chroma.sqlite3").is_file():
        store = load_vectorstore()
        store._collection.delete(where={"source": target.name})
        count = store._collection.count()
    target.unlink()
    progress = (
        _stepper_html(current=4, completed=(1, 2, 3))
        if count
        else _stepper_html()
    )
    return (
        f"Removed {target.name} from the document library and its indexed chunks.",
        _library_html(),
        _stats_html(count),
        progress,
    )


def _sources_html(documents):
    if not documents:
        return (
            '<div class="empty-state"><span class="empty-icon">⌕</span>'
            '<strong>No supporting passages found</strong>'
            '<span>Try a more specific question or check the selected source.</span></div>'
        )
    cards = []
    for document in documents:
        metadata = document.metadata or {}
        source = html.escape(str(metadata.get("source") or "Source not recorded"))
        page = metadata.get("page")
        chunk = metadata.get("chunk_id")
        details = []
        if page not in (None, "", 0):
            details.append(f"Page {html.escape(str(page))}")
        if chunk not in (None, ""):
            details.append(f"Chunk {html.escape(str(chunk))}")
        excerpt = html.escape(document.page_content.strip()[:520])
        cards.append(
            '<article class="source-card">'
            '<div class="source-heading">'
            f'<strong>{source}</strong><span>{html.escape(" · ".join(details))}</span>'
            '</div>'
            f'<p>{excerpt}</p></article>'
        )
    return '<div class="source-list">' + "".join(cards) + "</div>"


def ask(question, source_filter):
    if not question or not question.strip():
        return (
            "Enter a question to search your learning materials.",
            _sources_html([]),
            gr.skip(),
        )

    retriever = get_retriever(k=3, source=source_filter.strip() or None if source_filter else None)
    chain = build_chain(retriever)
    answer = chain.invoke(question.strip())
    documents = retriever.invoke(question.strip())
    return answer, _sources_html(documents), _stepper_html(current=5, completed=(1, 2, 3, 4))


def clear_chat():
    return "", "", ""


CSS = """
:root { --navy:#101b33; --blue:#172a4d; --gold:#c8a45d; --paper:#f7f8fa; --ink:#202938; }
body, .gradio-container { background:var(--paper)!important; color:var(--ink); font-family:"IBM Plex Sans","Source Sans 3","Segoe UI",sans-serif!important; }
.gradio-container { max-width:1440px!important; margin:0 auto!important; }
#hero { overflow:hidden; padding:0!important; margin:0 0 28px!important; border:0!important; border-radius:0 0 28px 28px!important; background:#101b33!important; box-shadow:0 16px 42px rgba(16,27,51,.16); }
.hero { position:relative; isolation:isolate; min-height:480px; display:flex; flex-direction:column; justify-content:space-between; padding:28px clamp(22px,6vw,88px) 48px; color:#fff; background:radial-gradient(ellipse at 75% 65%,rgba(200,164,93,.17),transparent 30%),linear-gradient(112deg,#101b33 5%,#142646 58%,#0d192e 100%); }
.hero-art { position:absolute; z-index:-1; inset:0 0 0 auto; width:min(73%,1000px); height:100%; opacity:.93; }
.hero-shade { position:absolute; z-index:-1; inset:0; background:linear-gradient(90deg,#101b33 0%,rgba(16,27,51,.97) 28%,rgba(16,27,51,.66) 55%,rgba(16,27,51,.08) 100%),linear-gradient(0deg,rgba(16,27,51,.45),transparent 42%); }
.ministry-row { display:flex; align-items:center; justify-content:space-between; gap:20px; border-bottom:1px solid rgba(255,255,255,.16); padding-bottom:18px; }
.ministry-en { font-size:12px; line-height:1.55; letter-spacing:.11em; text-transform:uppercase; font-weight:600; color:#f2eee5; max-width:380px; }
.ministry-ar { font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; text-align:right; direction:rtl; font-size:16px; font-weight:500; color:#f2eee5; }
.identity-note { color:#d5c293; font-size:10px; letter-spacing:.12em; text-transform:uppercase; white-space:nowrap; }
.hero-copy { max-width:680px; padding:44px 0 18px; }
.eyebrow { color:#dfc27e; font-size:12px; font-weight:600; letter-spacing:.2em; text-transform:uppercase; margin:0 0 14px; }
.hero h1 { margin:0; color:white; font-size:clamp(38px,5.3vw,68px); font-weight:700; letter-spacing:-.04em; line-height:1.05; max-width:670px; }
.hero-title-ar { margin:12px 0 18px; font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; color:#f2eee5; font-size:clamp(26px,3.8vw,42px); font-weight:600; line-height:1.55; }
.tagline { display:flex; flex-wrap:wrap; gap:7px 18px; align-items:center; color:#e7d8b8; font-weight:500; font-size:16px; }
.tagline-ar { direction:rtl; font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; }
.hero-description { max-width:560px; margin:15px 0 24px; color:#cfdae8; line-height:1.7; font-size:15px; }
.hero-cta { display:inline-flex; align-items:center; gap:14px; padding:13px 20px; border-radius:7px; background:var(--gold); color:#17233a!important; font-size:14px; font-weight:700; text-decoration:none!important; box-shadow:0 8px 24px rgba(200,164,93,.22); transition:transform .18s ease,background .18s ease; }
.hero-cta:hover { transform:translateY(-2px); background:#d8b875; }
.hero-foot { display:flex; justify-content:space-between; gap:20px; color:#aab8ca; font-size:11px; letter-spacing:.04em; }
.hero-foot span:last-child { direction:rtl; font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; }
#welcome, #workflow, #workspace, #library-card, #stats-card, #qa-card, #sources-card { border:1px solid #e4e8ee!important; border-radius:14px!important; background:#fff!important; box-shadow:0 6px 20px rgba(23,42,77,.045)!important; }
.welcome { padding:5px 2px 2px; }
.welcome-kicker { color:#a07b36; text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:700; margin-bottom:8px; }
.welcome h2 { margin:0; font-size:27px; color:var(--navy); letter-spacing:-.025em; }
.welcome-ar { margin:5px 0 7px; direction:rtl; text-align:left; font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; font-size:22px; font-weight:600; color:#293a59; }
.welcome p { color:#687487; margin:5px 0; line-height:1.65; }
.welcome .start-link { display:inline-flex; margin-top:10px; color:#8a682d; font-weight:700; text-decoration:none; border-bottom:1px solid #c8a45d; padding-bottom:3px; }
.section-heading { color:var(--navy); font-size:18px; font-weight:700; margin:0 0 4px; }
.section-sub { color:#738094; font-size:12px; margin:0 0 16px; }
.section-ar { font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; direction:rtl; color:#58677c; font-size:13px; margin-left:8px; font-weight:500; }
.stepper { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:0; padding:16px 8px 8px; }
.step { display:flex; gap:10px; align-items:flex-start; position:relative; padding:8px 10px 12px 4px; min-width:0; }
.step:not(:last-child):after { content:""; position:absolute; height:1px; background:#dfe4eb; left:42px; right:-8px; top:23px; z-index:0; }
.step-number { flex:0 0 31px; height:31px; border-radius:50%; display:grid; place-items:center; font-size:11px; font-weight:700; z-index:1; background:#eef1f5; color:#8994a4; border:1px solid #e4e8ee; }
.step.current .step-number { background:var(--gold); border-color:var(--gold); color:#15213a; box-shadow:0 0 0 4px rgba(200,164,93,.15); }
.step.complete .step-number { color:#fff; background:#295c54; border-color:#295c54; }
.step-copy { display:flex; flex-direction:column; gap:5px; padding-top:2px; min-width:0; }
.step-en { color:#26344b; font-size:11px; line-height:1.35; font-weight:600; }
.step-ar { color:#7b8797; font-family:"IBM Plex Sans Arabic","Noto Sans Arabic",Tahoma,sans-serif; font-size:11px; line-height:1.5; text-align:left; }
.step.current .step-en, .step.current .step-ar { color:#172a4d; }
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.stat-card { display:flex; flex-direction:column; gap:4px; background:#f6f8fb; border:1px solid #e8ecf1; padding:14px; border-radius:10px; }
.stat-value { font-size:25px; line-height:1.1; color:#172a4d; font-weight:700; }
.stat-label { color:#738094; font-size:11px; }
.stat-status { grid-column:1/-1; display:flex; flex-direction:row; align-items:center; color:#647286; font-size:11px; }
.status-dot { height:8px; width:8px; border-radius:50%; background:#c8a45d; }
.library-list { display:flex; flex-direction:column; gap:7px; }
.library-row { display:flex; align-items:center; gap:9px; border-bottom:1px solid #edf0f4; padding:8px 2px; }
.file-type { display:grid; place-items:center; flex:0 0 37px; height:29px; border-radius:5px; background:#f3eee2; color:#88692f; font-size:9px; font-weight:700; }
.file-name { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#354258; font-size:12px; }
.file-size { color:#8993a2; font-size:10px; white-space:nowrap; }
.empty-state { min-height:100px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; gap:7px; color:#7a8798; padding:15px; }
.empty-state strong { color:#354258; font-size:13px; }
.empty-state span:last-child { font-size:11px; line-height:1.5; }
.empty-icon { color:#bd9c5d; font-size:23px; }
.upload-summary { display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 8px; }
.upload-file { display:inline-flex; flex-direction:column; border:1px solid #e4e8ee; background:#f7f8fa; border-radius:7px; padding:7px 10px; font-size:11px; color:#344158; }
.upload-file small { color:#8791a0; margin-top:3px; }
.upload-note { color:#8791a0; font-size:11px; }
.validation-error { display:block; color:#a12e38; font-size:12px; padding:6px; }
.source-list { display:flex; flex-direction:column; gap:10px; }
.source-card { border:1px solid #e8ecf1; border-left:3px solid #c8a45d; border-radius:8px; background:#fbfcfd; padding:12px 14px; }
.source-heading { display:flex; align-items:center; flex-wrap:wrap; gap:7px; color:#243653; font-size:12px; }
.source-heading span { color:#8791a0; font-size:10px; }
.source-card p { color:#59677a; font-size:12px; line-height:1.65; margin:8px 0 0; white-space:pre-wrap; }
#answer-panel textarea { background:#fbfcfd!important; border-color:#e4e8ee!important; line-height:1.65!important; }
#status-message textarea { color:#42516a!important; font-size:12px!important; }
.gradio-container button.primary { background:#172a4d!important; border-color:#172a4d!important; }
.gradio-container button.primary:hover { background:#213b68!important; }
.gradio-container button.secondary { border-color:#dfe4eb!important; }
.gradio-container label, .gradio-container .label { color:#354258!important; }
.gradio-container input, .gradio-container textarea { border-color:#e2e7ed!important; }
.gradio-container input:focus, .gradio-container textarea:focus { border-color:#c8a45d!important; box-shadow:0 0 0 2px rgba(200,164,93,.13)!important; }
@media(max-width:900px) { .stepper { grid-template-columns:1fr; gap:3px; padding-left:2px; } .step { padding:5px 5px 7px; align-items:center; } .step:not(:last-child):after { height:auto; width:1px; left:19px; right:auto; top:39px; bottom:-9px; } .step-copy { flex-direction:row; flex-wrap:wrap; align-items:center; gap:5px 10px; } .step-ar { text-align:right; } .hero-art { width:100%; opacity:.65; } }
@media(max-width:620px) { .hero { min-height:490px; padding:20px 21px 26px; } .ministry-row { align-items:flex-start; flex-direction:column; gap:9px; } .ministry-ar { text-align:left; font-size:14px; } .identity-note { white-space:normal; } .hero-copy { padding:35px 0 18px; } .hero h1 { font-size:39px; } .hero-title-ar { font-size:27px; } .hero-foot { flex-direction:column; gap:7px; } .hero-foot span:last-child { text-align:left; direction:rtl; } .welcome h2 { font-size:23px; } }
"""


HERO = """
<section class="hero" aria-label="Egyptian AI Learning Assistant">
  <svg class="hero-art" viewBox="0 0 1050 520" preserveAspectRatio="xMidYMid slice" role="img" aria-label="Stylized Giza pyramids and Cairo skyline joined by a subtle digital network">
    <defs>
      <linearGradient id="sand" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#dfc27e"/><stop offset="1" stop-color="#876a3b"/></linearGradient>
      <linearGradient id="pyramid" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#d4b673" stop-opacity=".68"/><stop offset="1" stop-color="#6e5838" stop-opacity=".12"/></linearGradient>
      <linearGradient id="horizon" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#d6b66e" stop-opacity=".22"/><stop offset="1" stop-color="#101b33" stop-opacity="0"/></linearGradient>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <path d="M70 388 Q330 325 560 365 Q800 300 1080 360 L1080 520 70 520Z" fill="url(#horizon)"/>
    <path d="M98 422 L350 178 600 422Z" fill="url(#pyramid)" stroke="#d7bb79" stroke-opacity=".46" stroke-width="2"/>
    <path d="M365 422 L552 252 742 422Z" fill="url(#pyramid)" stroke="#d7bb79" stroke-opacity=".38" stroke-width="2"/>
    <path d="M98 422 L350 178 344 422Z" fill="#e0c47e" fill-opacity=".12"/>
    <path d="M365 422 L552 252 548 422Z" fill="#e0c47e" fill-opacity=".11"/>
    <g fill="#152542" stroke="#c6a65e" stroke-opacity=".24" stroke-width="1">
      <path d="M670 424v-72h28v72m8 0v-106h35v106m9 0v-61h24v61m8 0v-88h44v88m8 0v-119h29v119m8 0v-70h26v70m8 0v-94h37v94m8 0v-55h24v55m8 0v-131h28v131m8 0v-75h36v75"/>
      <path d="M679 351l5-9 5 9m35-33l6-11 6 11m67 18l7-12 7 12m45-31l7-13 7 13m79 17l7-12 7 12m43-44l6-10 6 10" fill="none"/>
    </g>
    <path d="M42 424 Q300 398 520 427 T1075 408" fill="none" stroke="#d0af65" stroke-opacity=".34"/>
    <g fill="none" stroke="#b8c9dd" stroke-opacity=".42" stroke-width="1.2">
      <path d="M406 134 L530 91 645 144 755 88 866 141 984 84"/>
      <path d="M478 228 L530 91 645 144 694 262 866 141 914 242 984 84"/>
      <path d="M530 91 L585 34 755 88 811 35 984 84"/>
      <path d="M645 144 L773 201 866 141 1004 188"/>
    </g>
    <g fill="#dfc27e" filter="url(#glow)">
      <circle cx="406" cy="134" r="3"/><circle cx="530" cy="91" r="4"/><circle cx="645" cy="144" r="3"/>
      <circle cx="755" cy="88" r="4"/><circle cx="866" cy="141" r="3"/><circle cx="984" cy="84" r="4"/>
      <circle cx="478" cy="228" r="3"/><circle cx="694" cy="262" r="3"/><circle cx="914" cy="242" r="3"/>
      <circle cx="585" cy="34" r="2.5"/><circle cx="811" cy="35" r="2.5"/><circle cx="773" cy="201" r="2.5"/>
    </g>
    <g fill="#f3dc9d" opacity=".8"><circle cx="710" cy="122" r="1.4"/><circle cx="925" cy="107" r="1.5"/><circle cx="590" cy="202" r="1.2"/><circle cx="807" cy="280" r="1.4"/><circle cx="1020" cy="291" r="1.5"/><circle cx="751" cy="40" r="1.2"/></g>
    <path d="M349 179 L350 173" stroke="#f4d786" stroke-width="2" filter="url(#glow)"/>
  </svg>
  <div class="hero-shade"></div>
  <div class="ministry-row">
    <div class="ministry-en">Ministry of Communications<br>and Information Technology</div>
    <div class="ministry-ar" lang="ar" dir="rtl">وزارة الاتصالات وتكنولوجيا المعلومات</div>
    <div class="identity-note">Proposed learning platform · Independent service</div>
  </div>
  <div class="hero-copy">
    <p class="eyebrow">Learning · Knowledge · Future</p>
    <h1>Egyptian AI<br>Learning Assistant</h1>
    <h2 class="hero-title-ar" lang="ar" dir="rtl">المساعد الذكي للتعلم المصري</h2>
    <div class="tagline"><span>Empowering Education Through Artificial Intelligence</span>
      <span class="tagline-ar" lang="ar" dir="rtl">نحو تعليم أكثر ذكاءً بالذكاء الاصطناعي</span></div>
    <p class="hero-description">A thoughtful learning workspace that turns your course materials into a searchable knowledge base, with answers grounded in your own documents.</p>
    <a class="hero-cta" href="#workspace">Launch AI Workspace <span aria-hidden="true">→</span>
      <span class="tagline-ar" lang="ar" dir="rtl">بدء مساحة العمل الذكية</span></a>
  </div>
  <div class="hero-foot"><span>Knowledge begins with a good question.</span><span lang="ar" dir="rtl">المعرفة تبدأ بسؤال جيد.</span></div>
</section>
"""


with gr.Blocks(title="Egyptian AI Learning Assistant") as demo:
    gr.HTML(HERO, elem_id="hero")
    with gr.Column():
        gr.HTML(
            '<div class="welcome"><div class="welcome-kicker">A focused space for curious minds</div>'
            '<h2>Welcome to your learning workspace</h2>'
            '<div class="welcome-ar" lang="ar" dir="rtl">مرحباً بك في مساحة التعلم الخاصة بك</div>'
            '<p>Bring your course notes, build a trusted knowledge base, and explore what you are learning—one question at a time.</p>'
            '<a class="start-link" href="#workspace">Start Learning <span lang="ar" dir="rtl"> &nbsp; ابدأ التعلم</span> ↗</a></div>'
        )

    with gr.Column(elem_id="workflow"):
        gr.HTML('<div class="section-heading">Your learning journey <span class="section-ar" lang="ar" dir="rtl">مسار التعلم</span></div>'
                '<div class="section-sub">Five connected steps, from your documents to verifiable answers.</div>')
        stepper = gr.HTML(_stepper_html())

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(elem_id="stats-card"):
                gr.HTML('<div class="section-heading">Knowledge base</div><div class="section-sub">Live document and index totals</div>')
                stats = gr.HTML(_stats_html())
            with gr.Column(elem_id="library-card"):
                gr.HTML('<div class="section-heading">Document library <span class="section-ar" lang="ar" dir="rtl">مكتبة المستندات</span></div>')
                library = gr.HTML(_library_html())
                remove_choice = gr.Textbox(label="Document filename to remove", placeholder="Enter the exact filename", elem_id="remove-choice")
                remove_button = gr.Button("Remove document", variant="secondary")
        with gr.Column(scale=2):
            with gr.Column(elem_id="workspace"):
                gr.HTML('<div class="section-heading">01 · Add learning materials <span class="section-ar" lang="ar" dir="rtl">رفع المستندات</span></div>'
                        '<div class="section-sub">Add one or more PDFs, Word documents or text files. Existing files are never silently overwritten.</div>')
                files = gr.File(
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath",
                    label="Choose learning materials",
                )
                upload_details = gr.HTML(_upload_summary([]))
                index_button = gr.Button("Build knowledge base", variant="primary")
                status = gr.Textbox(label="Processing status", interactive=False, lines=2, elem_id="status-message")

            with gr.Row():
                with gr.Column(scale=3, elem_id="qa-card"):
                    gr.HTML('<div class="section-heading">Ask your learning assistant <span class="section-ar" lang="ar" dir="rtl">اسأل مساعدك التعليمي</span></div>'
                            '<div class="section-sub">Answers are generated from your indexed materials. Ask a clear, focused question.</div>')
                    question = gr.Textbox(
                        label="Your question",
                        placeholder="For example: What are the main stages described in my notes?",
                        lines=2,
                    )
                    gr.Examples(
                        examples=[
                            ["What are the main ideas explained in these materials?"],
                            ["Summarize the key steps described in my notes."],
                            ["Which concepts should I review further?"],
                        ],
                        inputs=question,
                        label="Example questions · أمثلة للأسئلة",
                    )
                    source_filter = gr.Textbox(label="Optional source filter", placeholder="Exact filename, e.g. biology_notes.pdf")
                    with gr.Row():
                        ask_button = gr.Button("Ask a question", variant="primary")
                        copy_button = gr.Button("Copy answer", variant="secondary")
                        clear_button = gr.Button("Clear chat", variant="secondary")
                    answer = gr.Textbox(label="AI answer", lines=7, interactive=False, elem_id="answer-panel")
                with gr.Column(scale=2, elem_id="sources-card"):
                    gr.HTML('<div class="section-heading">Sources & verification <span class="section-ar" lang="ar" dir="rtl">المصادر والتحقق</span></div>'
                            '<div class="section-sub">Retrieved passages that support the response.</div>')
                    sources = gr.HTML(_sources_html([]))

    files.change(_upload_summary, inputs=files, outputs=upload_details, show_progress="hidden")
    index_button.click(
        index_files,
        inputs=files,
        outputs=[status, library, stats, stepper],
        show_progress="minimal",
    )
    remove_button.click(
        remove_document,
        inputs=remove_choice,
        outputs=[status, library, stats, stepper],
        show_progress="minimal",
    )
    ask_button.click(
        ask,
        inputs=[question, source_filter],
        outputs=[answer, sources, stepper],
        show_progress="minimal",
    )
    question.submit(
        ask,
        inputs=[question, source_filter],
        outputs=[answer, sources, stepper],
        show_progress="minimal",
    )
    clear_button.click(clear_chat, outputs=[question, answer, sources], show_progress="hidden")
    copy_button.click(
        fn=None,
        js="() => { const field = document.querySelector('#answer-panel textarea'); "
           "if (field && field.value) navigator.clipboard.writeText(field.value); }",
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
