"""Step 2: split text into overlapping chunks. Why? LLMs and search work best on
small focused passages, and overlap stops answers being cut in half."""
import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    id: int
    source: str
    page: int
    text: str

    def to_dict(self):
        return asdict(self)


def split_text(text: str, chunk_words: int = 120, overlap: int = 25) -> list[str]:
    """Pack whole paragraphs into chunks of ~chunk_words. The last paragraph of a chunk is
    repeated at the start of the next one if it is short (overlap), so answers aren't cut in half."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:                    # split any giant paragraph by words
        words = para.split()
        if len(words) <= chunk_words:
            units.append(para)
        else:
            units += [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    chunks, current, count = [], [], 0
    for unit in units:
        n = len(unit.split())
        if current and count + n > chunk_words:
            chunks.append("\n".join(current))
            last = current[-1]
            current, count = ([last], len(last.split())) if len(last.split()) <= overlap else ([], 0)
        current.append(unit)
        count += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_chunks(pages, chunk_words=120, overlap=25) -> list[Chunk]:
    out: list[Chunk] = []
    for source, page, text in pages:
        for piece in split_text(text, chunk_words, overlap):
            out.append(Chunk(len(out), source, page, piece))
    return out
