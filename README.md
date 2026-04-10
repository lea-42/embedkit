# embedkit

Extracts, chunks, and embeds PDF documents and queries for use in semantic search. 

## Pipeline

```
PDF → extract (pymupdf4llm) → chunk (markdown parser) → embed (sentence-transformers)
```

- **extract**: converts PDF to markdown, promotes numbered section headings to reflect depth, returns chunks with no side effects
- **chunk**: splits markdown into chunks with breadcrumb tracking, page numbers, and section context
- **embed**: embeds chunks using a multilingual model, streams `(chunk, embedding)` pairs — caller decides where to store them

## Library usage

```python
from embedkit.extractor import extract
from embedkit.embedder import embed_chunks, embed_texts, embed_query, load_model, E5_MULTILINGUAL_BASE

# Pre-warm at startup
await load_model(E5_MULTILINGUAL_BASE)

# Extract chunks from a PDF (no files written)
chunks = extract("policy.pdf")

# Embed chunks — defaults to the "text" key
for chunk, embedding in embed_chunks(chunks, config=E5_MULTILINGUAL_BASE):
    db.store(chunk, embedding)

# Include breadcrumbs for richer embeddings — list fields joined with \n, fields with \n\n
for chunk, embedding in embed_chunks(chunks, text_keys=["section_breadcrumbs", "text"]):
    db.store(chunk, embedding)

# Or embed raw texts directly
for embedding in embed_texts(["some text", "more text"], config=E5_MULTILINGUAL_BASE):
    db.store(embedding)

# At query time
query_embedding = embed_query("what is covered for trip cancellation?", config=E5_MULTILINGUAL_BASE)
```

## Example runner

Runs the full pipeline and saves `.md` and `_chunks.json` next to the PDF:

```bash
python example.py path/to/policy.pdf
```

## FastAPI startup

Load the model once at startup using the lifespan hook so it is ready before the first request:

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from embedkit.embedder import load_model, DEFAULT_MODEL, embed_query

@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/search")
async def search(q: str):
    query_embedding = embed_query(q)
    ...
```

## Model loading

By default `load_model()` downloads the model from HuggingFace on first call. For air-gapped environments (e.g. ECS without internet access), upload the model to S3 and set `MODEL_S3_PATH`.

### Upload a model to S3 (once)

```bash
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('multilingual-e5-base').save('/tmp/model')
"
aws s3 sync /tmp/model s3://my-bucket/models/multilingual-e5-base
```

### Load from S3 at runtime

```bash
export MODEL_S3_PATH=s3://my-bucket/models/multilingual-e5-base
```

`load_model()` will download from S3 to a local temp directory and load from there. No other code changes needed.

### Custom model

Pass any `SentenceTransformer`-compatible model via `EmbeddingModel`. You are responsible for prefixes.

```python
from embedkit.embedder import EmbeddingModel, load_model, embed_chunks

my_config = EmbeddingModel(
    model_name="sentence-transformers/all-mpnet-base-v2",
    query_prefix="",
    passage_prefix="",
)
await load_model(my_config)
for chunk, embedding in embed_chunks(chunks, config=my_config):
    ...
```

## Supported embedding models

| Constant | Model | Dims | Notes |
|---|---|---|---|
| `E5_MULTILINGUAL_BASE` (default) | `multilingual-e5-base` | 768 | Best quality, ~1.1GB |
| `E5_MULTILINGUAL_SMALL` | `multilingual-e5-small` | 384 | Faster, ~470MB |
| `MINILM_MULTILINGUAL` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | No prefix required |

E5 models require `"query: "` / `"passage: "` prefixes — handled automatically.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# production
pip install .

# development (includes pytest and type stubs)
pip install ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```
