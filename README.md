# nudify-elite (private cloud build)

Optimized local + RunPod/Tailscale build. Loads on CPU locally, CUDA on the pod.

## Pod setup (one paste, after cloning)
    export GH_TOKEN=... TS_AUTHKEY=... APP_USER=... APP_PASS=...
    bash pod_setup.sh

Then open https://nudify.<your-tailnet>.ts.net  (private — Tailscale only).
