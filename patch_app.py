#!/usr/bin/env python3
"""nudify-elite patcher: /models, /latest, model wiring, newest-only outputs. Idempotent."""
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
SRV, CORE, IDX = R / "server.py", R / "nudify" / "core.py", R / "static" / "index.html"

def patch(path, pairs):
    t = path.read_text(encoding="utf-8-sig")
    for marker, old, new in pairs:
        if marker in t:
            print("  =", path.name, "|", marker[:34]); continue
        if old not in t:
            print("  !! anchor missing:", path.name, "|", old[:44]); sys.exit(1)
        t = t.replace(old, new, 1); print("  +", path.name, "|", marker[:34])
    path.write_text(t, encoding="utf-8")

GP_OLD = '''def get_pipe(backend: str) -> NudifyPipeline:
    global _pipe, _pipe_backend
    if _pipe is None or _pipe_backend != backend:
        if _pipe is not None:
            del _pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _pipe = Nud
