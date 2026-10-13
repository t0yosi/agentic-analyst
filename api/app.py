"""FastAPI application exposing the analyst over HTTP."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from agents.orchestrator import Orchestrator
from api.schemas import AskRequest, AskResponse, HealthResponse
from config.settings import AppConfig, load_app_config
from database.connection import connect, table_counts
from llm.base import LLMClient
from llm.factory import build_llm_client
from utils.errors import AnalystError

logger = logging.getLogger(__name__)

_INDEX = """<!doctype html>
<title>Agentic Analyst</title>
<style>
 body{font:15px/1.55 system-ui,sans-serif;max-width:820px;margin:40px auto;padding:0 20px;
      background:#0f1419;color:#e6e9ee}
 h1{font-size:20px;margin-bottom:4px} p.sub{color:#8b95a5;margin-top:0}
 #log{margin:24px 0}
 .turn{padding:12px 14px;border-radius:8px;margin-bottom:10px}
 .you{background:#1c2530} .bot{background:#161b22;border:1px solid #262d38}
 .meta{color:#6b7684;font-size:12px;margin-top:6px}
 img{max-width:100%;border-radius:6px;margin-top:10px}
 form{display:flex;gap:8px} input{flex:1;padding:11px;border-radius:8px;
      border:1px solid #2a323d;background:#11161c;color:#e6e9ee}
 button{padding:11px 18px;border:0;border-radius:8px;background:#2b6cb0;color:#fff;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
</style>
<h1>Agentic Analyst</h1>
<p class="sub">Ask a question about the bikeshare network.</p>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Which borough has the longest average trip?" autocomplete="off">
<button id="b">Ask</button></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),
      q=document.getElementById('q'),b=document.getElementById('b');
function add(cls,html){const d=document.createElement('div');d.className='turn '+cls;
  d.innerHTML=html;log.appendChild(d);d.scrollIntoView({behavior:'smooth'});}
f.onsubmit=async e=>{e.preventDefault();const question=q.value.trim();if(!question)return;
  add('you',question.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])));
  q.value='';b.disabled=true;b.textContent='Thinking...';
  try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({question})});
    const j=await r.json();
    if(!r.ok){add('bot','<b>Error:</b> '+(j.detail||r.status));}
    else{let h=j.answer.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))
           .replace(/\\n/g,'<br>');
      if(j.chart_base64)h+='<img src="data:image/png;base64,'+j.chart_base64+'">';
      h+='<div class="meta">'+j.agents_called.join(', ')+' &middot; '+j.iterations+
         ' turns &middot; '+(j.input_tokens+j.output_tokens)+' tokens</div>';
      add('bot',h);}
  }catch(err){add('bot','<b>Error:</b> '+err.message);}
  b.disabled=false;b.textContent='Ask';q.focus();};
</script>
"""


def create_app(
    config: Optional[AppConfig] = None,
    client: Optional[LLMClient] = None,
) -> FastAPI:
    """Build the application.

    Args:
        config: Configuration to use. Loaded from the environment when omitted.
        client: Model client to use. Built from ``config`` when omitted, which
            lets tests inject a deterministic mock.

    Returns:
        A configured FastAPI instance.
    """
    app_config = config or load_app_config()
    llm_client = client or build_llm_client(app_config.llm)
    orchestrator = Orchestrator(llm_client, app_config)

    app = FastAPI(title="Agentic Analyst", version="1.0.0")
    app_config.log_active_config()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ready, counts = False, {}
        if app_config.database.path.exists():
            conn = connect(app_config.database.path, read_only=True)
            try:
                counts = table_counts(conn)
                ready = counts.get("trips", 0) > 0
            finally:
                conn.close()
        return HealthResponse(
            status="ok",
            model=llm_client.model_name,
            database_ready=ready,
            tables=counts,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = orchestrator.ask(request.question, remember=request.remember)
        except AnalystError as exc:
            logger.error("[STAGE] ask failed: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return AskResponse(
            answer=result.answer,
            chart_base64=result.chart_base64,
            agents_called=result.agents_called,
            iterations=result.iterations,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    @app.post("/reset")
    def reset() -> dict:
        orchestrator.reset_memory()
        return {"status": "memory cleared"}

    return app
