from __future__ import annotations
import argparse, os, uvicorn

def main() -> None:
    ap = argparse.ArgumentParser(prog="nudify-elite")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warm", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import torch
    print(f"[nudify-elite] device = {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if args.warm:
        from nudify.core import NudifyPipeline
        print("[nudify-elite] loading models...")
        NudifyPipeline()
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
