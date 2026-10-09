import base64, os
from starlette.responses import Response

def install(app):
    """HTTP Basic auth. Active only when APP_PASS is set."""
    user = os.getenv("APP_USER", "user")
    pw = os.getenv("APP_PASS", "")
    @app.middleware("http")
    async def _auth(request, call_next):
        if not pw:
            return await call_next(request)
        hdr = request.headers.get("authorization", "")
        if hdr.startswith("Basic "):
            try:
                u, p = base64.b64decode(hdr[6:]).decode().split(":", 1)
                if u == user and p == pw:
                    return await call_next(request)
            except Exception:
                pass
        return Response("Unauthorized", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="nudify"'})
