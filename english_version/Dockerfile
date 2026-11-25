FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Make your src directory the Python import root.
ENV PYTHONPATH=/app/src

ENV HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence_transformers \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3 available as python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Python dependencies
COPY requirements.txt /app/requirements.txt
COPY gliner_model/GLiNER /app/libs/GLiNER

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cu132


# source code and RAG datafiles
COPY src /app/src

COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh
RUN mkdir -p /app/index_cache

ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from gliner import GLiNER; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
GLiNER.from_pretrained('urchade/gliner_multi-v2.1')"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
