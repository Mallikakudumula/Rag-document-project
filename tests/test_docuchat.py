"""Run: python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docuchat.answer import NOT_FOUND, answer_question
from docuchat.chunking import build_chunks, split_text
from docuchat.config import Settings
from docuchat.retrieval import BM25, Retriever, rrf
from docuchat.server import make_handler
from docuchat.textutil import tokenize

DOCS = [("refund.txt", 1, "Customers can return items within 30 days of delivery for a full refund."),
        ("it.txt", 1, "Passwords must be at least 12 characters. MFA is mandatory for email."),
        ("leave.txt", 1, "Employees receive 18 days of paid annual leave per year.")]


def make_retriever():
    return Retriever(build_chunks(DOCS))


class ChunkingTests(unittest.TestCase):
    def test_respects_size_and_overlap(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
        chunks = split_text(text, chunk_words=80, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= 140 for c in chunks))
        self.assertIn("Paragraph 2", chunks[0] + chunks[1])

    def test_empty(self):
        self.assertEqual(split_text("   "), [])


class RetrievalTests(unittest.TestCase):
    def test_bm25_ranks_relevant_first(self):
        hits = make_retriever().search("how many days to return an item", 3)
        self.assertEqual(hits[0].chunk.source, "refund.txt")

    def test_rrf_merges_rankings(self):
        self.assertEqual(rrf([[1, 2, 3], [3, 1, 2]])[0], 1)

    def test_tokenizer_drops_stopwords_and_stems(self):
        self.assertEqual(tokenize("The refunds are"), ["refund"])


class AnswerTests(unittest.TestCase):
    s = Settings(min_score=0.3)   # tiny 3-chunk corpus => lower BM25 scores than real data

    def test_answers_with_citation(self):
        a = answer_question("What is the return window?", make_retriever(), None, self.s)
        self.assertTrue(a.found)
        self.assertIn("30 days", a.answer)
        self.assertEqual(a.sources[0]["source"], "refund.txt")

    def test_refuses_unrelated_question(self):
        a = answer_question("What is the CEO's favourite colour?", make_retriever(), None, self.s)
        self.assertFalse(a.found)
        self.assertEqual(a.answer, NOT_FOUND)

    def test_llm_path_uses_context_and_prompt(self):
        class FakeLLM:
            system = user = ""
            def complete(self, system, user, max_tokens=600):
                from docuchat.llm import LLMResponse
                self.system, self.user = system, user
                return LLMResponse("30 days [1]", 100, 5)
        llm = FakeLLM()
        a = answer_question("What is the return window?", make_retriever(), llm, self.s)
        self.assertIn("refund.txt", llm.user)
        self.assertIn("ONLY", llm.system)
        self.assertEqual((a.input_tokens, a.output_tokens), (100, 5))


class ServerTests(unittest.TestCase):
    def test_api(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(make_retriever(), None, Settings(min_score=0.3)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        req = urllib.request.Request(base + "/api/ask", json.dumps({"question": "return window?"}).encode(),
                                     {"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        self.assertTrue(data["found"])
        bad = urllib.request.Request(base + "/api/ask", b"{}", {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(bad)
        self.assertEqual(ctx.exception.code, 400)
        server.shutdown()


if __name__ == "__main__":
    unittest.main()
