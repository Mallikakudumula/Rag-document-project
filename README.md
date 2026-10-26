# DocuChat: RAG Question Answering over Company Documents

Ask questions in plain English about your documents (PDF, Word, Markdown, text) and get answers
**with source citations**. If the answer isn't in the documents, it says so instead of making things up.

> Sample documents are **fake** (a made-up company, "Acme Retail"). Never commit real company documents.

**Why this is a good portfolio / freelance project:** "chat with our documents" is one of the most
requested AI services for small businesses (HR policies, product manuals, SOPs, contracts, support
knowledge bases). This repo shows the full RAG pipeline *and* how to measure it.

## How RAG works (the 4 steps, each is one file)

```
 documents ──► 1 LOAD ──► 2 CHUNK ──► 3 INDEX/RETRIEVE ──► 4 ANSWER (with citations)
              loaders.py  chunking.py  retrieval.py/index.py   answer.py + llm.py
```

1. **Load** text from PDF/DOCX/MD/TXT (`loaders.py`).
2. **Chunk** into small overlapping passages (`chunking.py`), because search and LLMs work best on focused text.
3. **Retrieve** the best chunks for a question (`retrieval.py`): BM25 keyword search by default, optional
   semantic embeddings, merged with **Reciprocal Rank Fusion** (hybrid search).
4. **Answer** using only those chunks (`answer.py`). A real LLM writes the answer; the offline mode quotes the best sentences.

### Guardrails (what makes it production-minded)
- **"Not found" check** before calling the LLM (score + question-term coverage), so it doesn't guess.
- **Strict prompt:** answer only from context, cite `[n]`, refuse otherwise.
- **Prompt-injection defence:** retrieved text is declared untrusted data in the system prompt.
- **Evaluation** (`eval/eval_set.json`): hit@k, MRR and refusal accuracy, so changes are measured, not guessed.

## Quick start (no API key, no internet needed)

```powershell
cd "E:\company projects\docuchat-rag"
python -m venv .venv
.venv\Scripts\activate
pip install python-dotenv requests pypdf python-docx
$env:PYTHONPATH="src"          # Mac/Linux: export PYTHONPATH=src

python -m docuchat ingest
python -m docuchat ask "How many days of annual leave do I get?"
python -m docuchat ask "What is the pet policy for the office?"    # -> not found (correct!)
python -m docuchat eval
python -m docuchat serve --port 8000                               # open http://127.0.0.1:8000
python -m unittest discover -s tests -v
```

Use your own documents: put files in `data/docs/` and run `ingest` again.

## Use a real LLM
Copy `.env.example` to `.env`, then set one of:
- `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` (model in `LLM_MODEL`)
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY` (also works with Azure OpenAI / Ollama via `OPENAI_BASE_URL`)

The real-LLM path is covered by a unit test with a fake LLM, but I have **not** run it against live APIs.
Test with your own key before demoing it.

## Honest limitation (a great interview talking point)
Run `python -m docuchat eval`: 14/15 pass. The failure, "What is the minimum password length?", fails because the
policy says "at least 12 characters long": different words, same meaning. **Keyword search can't bridge that.**
Enable semantic search (`pip install sentence-transformers numpy`, set `USE_EMBEDDINGS=true`, re-run `ingest`)
to fix this class of failure. That path is implemented but untested here, so verify it on your machine.

## Project structure
```
src/docuchat/  config, loaders, chunking, textutil, retrieval, index, llm, answer, evaluate, server, cli
data/docs/     fake sample documents (replace with your own)
eval/          eval_set.json (questions + expected source, incl. unanswerable ones)
tests/         9 unit tests
```

## Ideas to extend (and sell)
- Swap the JSON index for a vector DB (Azure AI Search, pgvector, Qdrant)
- Add a reranker, conversation memory, and per-user document permissions
- Streamlit/React front-end, Docker image, deploy to Azure App Service
- Offer it as a paid service: "private AI assistant for your company handbook"
