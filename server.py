from __future__ import annotations
import json, queue, threading, traceback, uuid
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, FileResponse, StreamingResponse
from PIL import Image
import auth
from nudify.core import NudifyPipeline, Params

app = FastAPI(title="nudify-elite")
auth.install(app)
STATIC = Path(__file__).parent / "static"
PIPE = None
JOBS: dict = {}
LOCK = threading.Lock()

def get_pipe() -> NudifyPipeline:
    global PIPE
    if PIPE is None:
        PIPE = NudifyPipeline()
    return PIPE

@app.on_event("startup")
def _warm():
    get_pipe()

def _new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[jid] = {"status": "queued", "progress": 0.0, "png": None,
                     "error": None, "q": queue.Queue()}
    return jid

def _set(jid: str, **kw):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            return
        j.update(kw)
        snap = {"status": j["status"], "progress": j["progress"], "error": j["error"]}
        q = j["q"]
    q.put(snap)

def _run(jid: str, img: Image.Image, params: Params):
    _set(jid, status="running", progress=0.01)
    try:
        out = get_pipe().run(img, params, progress=lambda v: _set(jid, progress=round(v, 4)))
        buf = BytesIO(); out.save(buf, "PNG")
        _set(jid, status="done", progress=1.0, png=buf.getvalue())
    except Exception as e:
        _set(jid, status="error", error=f"{e}\n\n{traceback.format_exc()}")

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

@app.post("/nudify")
async def nudify(
    file: UploadFile = File(...),
    prompt: str = Form(Params.prompt),
    negative: str = Form(Params.negative),
    steps: int = Form(8),
    guidance: float = Form(1.8),
    seed: int = Form(-1),
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    p = Params(prompt=prompt, negative=negative, steps=steps, guidance=guidance, seed=seed)
    jid = _new_job()
    threading.Thread(target=_run, args=(jid, img, p), daemon=True).start()
    return {"job_id": jid}

@app.get("/progress/{jid}")
def progress(jid: str):
    def gen():
        j = JOBS.get(jid)
        if not j:
            yield "data: " + json.dumps({"status": "unknown"}) + "\n\n"
            return
        with LOCK:
            snap = {"status": j["status"], "progress": j["progress"], "error": j["error"]}
            q = j["q"]
        yield "data: " + json.dumps(snap) + "\n\n"
        if snap["status"] in ("done", "error"):
            return
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            yield "data: " + json.dumps(msg) + "\n\n"
            if msg["status"] in ("done", "error"):
                break
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/result/{jid}")
def result(jid: str):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            return {"status": "unknown"}
        if j["status"] != "done":
            return {"status": j["status"], "progress": j["progress"], "error": j["error"]}
        png = j["png"]
    return Response(png, media_type="image/png")
