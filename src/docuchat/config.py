import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    provider: str = "mock"
    anthropic_key: str = ""
    model: str = "claude-sonnet-5-5"
    openai_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    docs_dir: Path = Path("data/docs")
    index_path: Path = Path("data/index.json")
    chunk_words: int = 120
    chunk_overlap: int = 25
    top_k: int = 4
    min_score: float = 1.0        # BM25 score the best chunk must reach
    min_coverage: float = 0.34     # share of question terms that must appear in best chunk
    use_embeddings: bool = False


def load_settings() -> Settings:
    g = os.getenv
    return Settings(
        provider=g("LLM_PROVIDER", "mock").lower(),
        anthropic_key=g("ANTHROPIC_API_KEY", ""), model=g("LLM_MODEL", "claude-sonnet-5-5"),
        openai_key=g("OPENAI_API_KEY", ""), openai_base_url=g("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=g("OPENAI_MODEL", "gpt-4o-mini"),
        docs_dir=Path(g("DOCS_DIR", "data/docs")), index_path=Path(g("INDEX_PATH", "data/index.json")),
        chunk_words=int(g("CHUNK_WORDS", "120")), chunk_overlap=int(g("CHUNK_OVERLAP", "25")),
        top_k=int(g("TOP_K", "4")), min_score=float(g("MIN_SCORE", "1.0")),
        min_coverage=float(g("MIN_COVERAGE", "0.34")),
        use_embeddings=g("USE_EMBEDDINGS", "false").lower() == "true",
    )
