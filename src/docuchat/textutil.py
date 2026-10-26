import re

STOPWORDS = set("""a an the is are was were be been am do does did of to in on at for from by with and or
but if then so as it its this that these those i you we they he she my our your their what which who whom
how when where why can could should would will shall may might must not no any all per about into than
many much long get take need does company provide tell""".split())


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = [re.sub(r"^#+\s*", "", p).strip() for p in parts]
    return [p for p in cleaned if len(p) > 15]
