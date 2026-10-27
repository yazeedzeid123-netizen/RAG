"""STEP 1 - LOAD: read files and keep metadata using `unstructured`."""
from pathlib import Path
from unstructured.partition.auto import partition
from config import DATA_DIR


def load_elements(folder=DATA_DIR):
    """Return a list of (filename, elements). Each element = Title / NarrativeText / Table ..."""
    results = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() not in {".txt", ".pdf", ".docx", ".md", ".html"}:
            continue
        kwargs = {"strategy": "fast"} if path.suffix.lower() == ".pdf" else {}
        elements = partition(filename=str(path), **kwargs)
        results.append((path.name, elements))
    return results


if __name__ == "__main__":
    for name, elements in load_elements():
        print(f"\n=== {name}: {len(elements)} elements ===")
        for el in elements[:6]:
            print(f"[{el.category}] {el.text[:70]!r}")
            print("   metadata:", {k: v for k, v in el.metadata.to_dict().items()
                                   if k in ("filename", "page_number", "filetype")})
