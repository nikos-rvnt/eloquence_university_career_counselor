# ELOQUENCE

# CNR Pilot — University Career Counselor — Docker Setup

This project implements a university career-counseling dialogue system using SDialog, GLiNER, Ollama/Gemma, Sentence-Transformers, Cross-Encoder reranking, and FAISS-based RAG over the university `.txt` files.

## 1. Project structure

```text
project/
├── Dockerfile
├── docker-compose.yaml
├── docker-entrypoint.sh
├── requirements.txt
├── index_cache/
└── src/
    ├── university_counselor.py
    ├── build_index.py
    ├── config.py
    ├── models.py
    ├── extraction.py
    ├── normalization.py
    ├── llm.py
    ├── rag/
    │   ├── __init__.py
    │   ├── documents.py
    │   ├── indexing.py
    │   ├── retrieval.py
    │   └── recommendations.py
    └── italian_universities/
        └── *.txt
```

## 2. Main dependencies

Required infrastructure:

- Docker Engine
- Docker Compose (`docker compose`)
- NVIDIA Container Toolkit when CUDA/GPU execution is used

The Dockerfile installs the project's `requirements.txt` and then explicitly installs a CUDA 13.2 PyTorch - this should be replaced with your actual CUDA version:

The Dockerfile also copies the local GLiNER implementation/package from:

```text
gliner_model/GLiNER
```

to:

```text
/app/libs/GLiNER
```

The Python image depends on both `requirements.txt` and the local `gliner_model/GLiNER` directory.


## 3. Runtime architecture

```text
┌───────────────────────────────────────────┐
│ Application container                     │
│                                           │
│ university_counselor.py / SDialog         │
│ profile extraction / GLiNER + FewShot LLM │
│ FAISS / RAG                               │
└──────────────┬────────────────────────────┘
               │ HTTP
               ▼
┌──────────────────────────────┐
│ Ollama container             │
│ Gemma LLM                    │
└──────────────────────────────┘
```

Inside Docker, the application should reach Ollama through the Compose service name, normally:

```text
http://ollama:11434
```

Do not use `localhost` from the application container to reach the Ollama container.

## 4. Ollama configuration

Set the application environment to:

```yaml
environment:
  OLLAMA_BASE_URL: http://ollama:11434
```

The configured model must match the model installed in Ollama, for example:

```python
MODEL_URI = "gemma4:26b"
```

If your GPU's VRAM allows it, use a bigger Gemma4 version (31B).

Check installed models with:

```bash
docker compose exec ollama ollama list
```

Pull the configured model when necessary:

```bash
docker compose exec ollama ollama pull gemma4:26b
```

If the host exposes Ollama on another port, that is only the host-side port. Container-to-container communication still uses `ollama:11434`.

## 5. Build the Docker images

From the project root:

```bash
docker compose build
```

For a clean rebuild:

```bash
docker compose build --no-cache
```

## 6. Start the services

Start the whole Compose stack:

```bash
docker compose up
```

Or in detached mode:

```bash
docker compose up -d
```

Follow logs with:

```bash
docker compose logs -f university-counselor-app
```

and:

```bash
docker compose logs -f ollama
```

Use the actual service names defined in `docker-compose.yaml` if they differ from `app` and `ollama`.

## 7. Build/load the FAISS RAG cache

The Docker entrypoint automatically executes:

```bash
python -u /app/src/build_index.py
```

before starting the interactive counselor. Therefore, you normally do **not** need a separate cache-building command.

The startup sequence inside the application container is:

```text
container starts
      ↓
docker-entrypoint.sh
      ↓
/app/src/build_index.py
      ↓
FAISS index is built or loaded
      ↓
/app/src/university_counselor.py
```

Rebuild the Docker image when the Python indexing code or dependencies change:

```bash
docker compose build
```

Rebuild the RAG cache when the university source files, parsing/chunking logic, metadata extraction, embedding model, or indexing logic changes. The exact cache behavior is controlled by `build_index.py` and the current index-signature configuration.

Because the entrypoint runs `build_index.py` automatically, restarting/recreating the application container triggers the cache build/load step:

```bash
docker compose up --force-recreate app
```

## 8. Run the interactive counselor

The current application is an interactive CLI, not a FastAPI endpoint.

The `docker-entrypoint.sh` script automatically starts:

```bash
python -u /app/src/university_counselor.py
```

Therefore, start the application service with:

```bash
docker compose up university-counselor-app
```

For an interactive foreground session, this is the preferred command because the container receives your terminal input directly.

A normal session should eventually reach:

```text
Counselor: ... profile question ...
User: ...
Counselor: ...
User: ...
Counselor: Here are the recommended university programmes: ...
```

## 9. Recommended startup sequence

For a first run:

```bash
docker compose build

docker compose up -d ollama
docker compose exec ollama ollama list
docker compose exec ollama ollama pull gemma4:26b
docker compose up university-counselor-app
```

The final command automatically runs the application's entrypoint, which first executes:

```bash
python -u /app/src/build_index.py
```

and then:

```bash
python -u /app/src/university_counselor.py
```


## 10. GPU requirements

The current retrieval implementation initializes the embedding model and Cross-Encoder on CUDA, so GPU execution is expected for that configuration. A GPU deployment therefore normally requires:

- NVIDIA GPU
- compatible NVIDIA driver
- NVIDIA Container Toolkit
- Docker GPU access configured by the Compose file

Ollama can also use the GPU when the Ollama service is configured for it.

## 11. University RAG data

Source documents are stored under:

```text
src/italian_universities/
```

The source `.txt` files are treated as read-only knowledge sources. The indexing layer extracts programme metadata such as:

```text
UNIVERSITY
COURSE
COURSE_CODE
DEGREE / TYPE
CURRICULUM
SECTION
URL
REGION
```

After source or indexing changes, rebuild the FAISS cache before starting the application.

## 12. Common problems

### `ollama: executable file not found`

The application container should normally connect to the separate Ollama service rather than trying to execute a local `ollama` binary. Verify:

```text
OLLAMA_BASE_URL=http://ollama:11434
```

### `Connection refused`

Check:

```bash
docker compose ps
docker compose logs ollama
```

and verify the application uses the internal Ollama address `http://ollama:11434`.

### `invalid model name`

Compare:

```bash
docker compose exec ollama ollama list
```

with the exact value of `config.MODEL_URI`.

### FAISS cache not found

The application normally builds/loads the RAG cache automatically during startup through:

```bash
python -u /app/src/build_index.py
```

Check the application logs:

```bash
docker compose logs -f university-counselor-app
```

Then recreate the application container if necessary:

```bash
docker compose up --force-recreate university-counselor-app
```

### CUDA/GPU errors

Check the host with:

```bash
nvidia-smi
```

and verify that the application container has GPU access.

## 13. Clean shutdown

Stop the stack:

```bash
docker compose down
```

For a clean rebuild:

```bash
docker compose down --remove-orphans
docker compose build --no-cache
```

Do not remove `index_cache/` unless you intentionally want to rebuild the RAG index.

## 14. Minimal command sequence

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama list
docker compose exec ollama ollama pull gemma4:26b
docker compose up university-counselor-app (or docker compose run --rm university-counselor-app)
```

At application startup, `docker-entrypoint.sh` automatically:

1. runs `/app/src/build_index.py`
2. starts `/app/src/university_counselor.py`

Then interact with the counselor through the CLI.
