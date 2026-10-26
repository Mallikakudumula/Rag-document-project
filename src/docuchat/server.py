"""Tiny web demo using only the standard library: python -m docuchat serve"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .answer import answer_question
from .index import load_retriever
from .llm import build_llm

PAGE = """<!doctype html><meta charset=utf-8><title>DocuChat</title>
<style>body{font:16px system-ui;max-width:720px;margin:40px auto;padding:0 16px}
input{width:75%;padding:10px}button{padding:10px 16px}.src{color:#555;font-size:14px;margin:6px 0}
#out{margin-top:20px;white-space:pre-wrap}</style>
<h2>DocuChat: ask your documents</h2>
<input id=q placeholder="e.g. How many days do I have to return an item?" autofocus>
<button onclick=ask()>Ask</button><div id=out></div>
<script>
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
async function ask(){
 const out=document.getElementById('out');out.textContent='Thinking...';
 const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({question:document.getElementById('q').value})});
 const d=await r.json();out.textContent='';
 const a=document.createElement('p');a.textContent=d.answer;out.appendChild(a);
 (d.sources||[]).forEach(s=>{const e=document.createElement('div');e.className='src';
  e.textContent='['+s.n+'] '+s.source+' (p.'+s.page+') - '+s.snippet+'...';out.appendChild(e);});
}
</script>"""


def make_handler(retriever, llm, settings):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, '{"status":"ok"}')
            else:
                self._send(404, '{"error":"not found"}')

        def do_POST(self):
            if self.path != "/api/ask":
                return self._send(404, '{"error":"not found"}')
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                question = str(body["question"]).strip()[:500]
                if not question:
                    raise KeyError
            except (ValueError, KeyError):
                return self._send(400, '{"error":"send JSON like {\\"question\\": \\"...\\"}"}')
            a = answer_question(question, retriever, llm, settings)
            self._send(200, json.dumps({"answer": a.answer, "found": a.found, "sources": a.sources}))

        def log_message(self, *args):   # keep console quiet
            pass
    return Handler


def serve(settings, port: int = 8000):
    retriever = load_retriever(settings)
    server = HTTPServer(("127.0.0.1", port), make_handler(retriever, build_llm(settings), settings))
    print(f"DocuChat running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    server.serve_forever()
