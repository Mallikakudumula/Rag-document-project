"""Measure retrieval quality instead of guessing. Metrics:
hit@k = was the expected document in the top k?   MRR = how high was it ranked?
refusal accuracy = did we correctly say 'not found' for unanswerable questions?"""
import json
from pathlib import Path

from .answer import is_answerable


def run_eval(retriever, settings, eval_path: Path) -> dict:
    cases = json.loads(Path(eval_path).read_text())
    hits_ok, rr, refusal_ok, n_ans, n_unans, details = 0, 0.0, 0, 0, 0, []
    for c in cases:
        hits = retriever.search(c["question"], settings.top_k)
        answerable = is_answerable(hits, settings.min_score, settings.min_coverage)
        if c.get("expected_source"):
            n_ans += 1
            sources = [h.chunk.source for h in hits]
            if c["expected_source"] in sources:
                hits_ok += 1
                rr += 1 / (sources.index(c["expected_source"]) + 1)
            ok = answerable and c["expected_source"] in sources
        else:
            n_unans += 1
            ok = not answerable
            refusal_ok += ok
        details.append((ok, c["question"]))
    return {"questions": len(cases), "hit_at_k": hits_ok / max(n_ans, 1), "mrr": rr / max(n_ans, 1),
            "refusal_accuracy": refusal_ok / max(n_unans, 1), "details": details}
