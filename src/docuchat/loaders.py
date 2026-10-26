"""Step 1 of RAG: read documents. Returns [(page_number, text)] per file."""
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def load_file(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == ".docx":
        import docx
        d = docx.Document(str(path))
        return [(1, "\n\n".join(p.text for p in d.paragraphs if p.text.strip()))]
    raise ValueError(f"Unsupported file type: {ext}")


def load_folder(folder: Path):
    """Yield (source_name, page, text) for every supported file."""
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            for page, text in load_file(path):
                if text.strip():
                    yield path.name, page, text
