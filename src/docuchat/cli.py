"""  python -m docuchat ingest
     python -m docuchat ask "How many days of annual leave do I get?"
     python -m docuchat chat
     python -m docuchat eval
     python -m docuchat serve --port 8000"""
import argparse
import logging
from pathlib import Path

from .answer import answer_question
from .config import load_settings
from .evaluate import run_eval
from .index import build_index, load_retriever
from .llm import build_llm


def print_answer(a):
    print(f"\n{a.answer}")
    for s in a.sources:
        print(f"   [{s['n']}] {s['source']} p.{s['page']} (score {s['score']})")
    if a.input_tokens:
        print(f"   tokens in/out: {a.input_tokens}/{a.output_tokens}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="docuchat")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask"); a.add_argument("question")
    sub.add_parser("chat")
    e = sub.add_parser("eval"); e.add_argument("--file", default="eval/eval_set.json")
    sv = sub.add_parser("serve"); sv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = load_settings()
    if args.cmd == "ingest":
        print(f"Indexed {build_index(s)} chunks -> {s.index_path}")
    elif args.cmd == "ask":
        print_answer(answer_question(args.question, load_retriever(s), build_llm(s), s))
    elif args.cmd == "chat":
        r, llm = load_retriever(s), build_llm(s)
        print("Ask about your documents (blank line to quit)")
        while (q := input("\n> ").strip()):
            print_answer(answer_question(q, r, llm, s))
    elif args.cmd == "eval":
        res = run_eval(load_retriever(s), s, Path(args.file))
        for ok, q in res["details"]:
            print(("PASS  " if ok else "FAIL  ") + q)
        print(f"\nquestions={res['questions']}  hit@{s.top_k}={res['hit_at_k']:.2f}  "
              f"MRR={res['mrr']:.2f}  refusal_accuracy={res['refusal_accuracy']:.2f}")
    elif args.cmd == "serve":
        from .server import serve
        serve(s, args.port)


if __name__ == "__main__":
    main()
