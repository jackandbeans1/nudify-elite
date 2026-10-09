FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/workspace/hf
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
EXPOSE 8000
CMD ["python","run.py","--host","0.0.0.0","--port","8000","--warm"]
