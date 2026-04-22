import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

from docvec.logging_config import log_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingModel:
    model_name: str
    query_prefix: str
    passage_prefix: str


E5_MULTILINGUAL_BASE = EmbeddingModel(
    model_name="multilingual-e5-base",
    query_prefix="query: ",
    passage_prefix="passage: ",
)

E5_MULTILINGUAL_SMALL = EmbeddingModel(
    model_name="multilingual-e5-small",
    query_prefix="query: ",
    passage_prefix="passage: ",
)

MINILM_MULTILINGUAL = EmbeddingModel(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    query_prefix="",
    passage_prefix="",
)

SUPPORTED_MODELS = (E5_MULTILINGUAL_BASE, E5_MULTILINGUAL_SMALL, MINILM_MULTILINGUAL)

DEFAULT_MODEL = E5_MULTILINGUAL_BASE

_model: SentenceTransformer | None = None
_model_config: EmbeddingModel | None = None
_load_lock = asyncio.Lock()


_S3_RETRY_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})


def _download_from_s3(s3_path: str) -> str:
    """Download a model directory from S3 to a local temp dir, return the local path.

    s3_path format: s3://bucket/prefix
    Retries up to 5 times with adaptive backoff on transient errors.
    """
    s3_path = s3_path.rstrip("/")
    bucket, _, prefix = s3_path.removeprefix("s3://").partition("/")
    local_dir = Path(tempfile.mkdtemp()) / "model"
    local_dir.mkdir()

    s3 = boto3.client("s3", config=_S3_RETRY_CONFIG)
    paginator = s3.get_paginator("list_objects_v2")

    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]
    if not keys:
        raise RuntimeError(f"No files found at {s3_path}")

    logger.info("downloading %d file(s) from %s", len(keys), s3_path)
    for key in keys:
        relative = key[len(prefix):].lstrip("/")
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, key, str(dest))
            logger.debug("downloaded %s", key)
        except Exception:
            logger.exception("failed to download %s", key)
            raise

    logger.info("download complete: %d file(s) saved to %s", len(keys), local_dir)
    return str(local_dir)


@log_time(logger)
async def load_model(config: EmbeddingModel = DEFAULT_MODEL) -> None:
    """Load the model at startup. Safe to call concurrently — loads only once.

    If called a second time (e.g. concurrently at startup), the second call is a no-op
    and the config argument is ignored. Only one model is supported at a time.

    If MODEL_S3_PATH is set (e.g. s3://my-bucket/models/multilingual-e5-base),
    the model is downloaded from S3 before loading. Otherwise loads from HuggingFace.
    """
    global _model, _model_config
    async with _load_lock:
        if _model is not None:
            return
        s3_path = os.environ.get("MODEL_S3_PATH")
        if s3_path:
            bucket_and_prefix = s3_path.removeprefix("s3://")
            if not s3_path.startswith("s3://") or "/" not in bucket_and_prefix or bucket_and_prefix.endswith("/"):
                raise ValueError(f"MODEL_S3_PATH must be in the form s3://bucket/prefix, got: {s3_path!r}")
            logger.info("downloading model from %s", s3_path)
            model_path = await asyncio.to_thread(_download_from_s3, s3_path)
        else:
            model_path = config.model_name
        logger.info("loading model %s", config.model_name)
        _model = await asyncio.to_thread(SentenceTransformer, model_path)
        _model_config = config
        if s3_path:
            shutil.rmtree(model_path, ignore_errors=True)
            logger.debug("cleaned up temp model dir %s", model_path)


def _get_model(config: EmbeddingModel) -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup before use.")
    if _model_config != config:
        raise RuntimeError(
            f"Requested model '{config.model_name}' but loaded model is '{_model_config.model_name}'. "
            "Only one model is supported at a time."
        )
    return _model


def embed_texts(
    texts: list[str],
    config: EmbeddingModel = DEFAULT_MODEL,
    model: SentenceTransformer | None = None,
) -> Generator[np.ndarray, None, None]:
    """Embed a list of texts and yield embeddings.

    Encodes all texts in a single batched call.
    If model is provided, it is used as-is — caller is responsible for any prefixes.
    Otherwise uses the singleton loaded via load_model(), applying the passage prefix from config.
    """
    if not texts:
        return

    if model is not None:
        resolved_model = model
        prefixed = texts
    else:
        resolved_model = _get_model(config)
        prefixed = [f"{config.passage_prefix}{t}" for t in texts]

    embeddings: np.ndarray = resolved_model.encode(prefixed, show_progress_bar=False)
    yield from embeddings


def _chunk_to_text(chunk: dict, text_keys: list[str]) -> str:
    parts = []
    for key in text_keys:
        value = chunk[key]
        parts.append("\n".join(value) if isinstance(value, list) else value)
    return "\n\n".join(parts)


def embed_chunks(
    chunks: list[dict],
    text_keys: list[str] | None = None,
    config: EmbeddingModel = DEFAULT_MODEL,
    model: SentenceTransformer | None = None,
) -> Generator[tuple[dict, np.ndarray], None, None]:
    """Embed chunks and yield (chunk, embedding) pairs.

    Each chunk must contain all keys in text_keys. Fields that are lists are joined
    with newline; fields are joined to each other with double newline before embedding.
    Defaults to text_keys=["text"].

    Example:
        embed_chunks(chunks, text_keys=["section_breadcrumbs", "text"])
        # embeds: "Intro\\nBenefits\\n\\nsome chunk text"

    Delegates to embed_texts for encoding.
    If model is provided, it is used as-is — caller is responsible for any prefixes.
    Otherwise uses the singleton loaded via load_model(), with prefixes from config.
    Caller decides what to do with the embeddings.
    """
    if not chunks:
        return

    texts = [_chunk_to_text(c, text_keys or ["text"]) for c in chunks]
    for chunk, embedding in zip(chunks, embed_texts(texts, config=config, model=model)):
        yield chunk, embedding


def embed_query(
    query: str,
    config: EmbeddingModel = DEFAULT_MODEL,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """Embed a search query.

    If model is provided, it is used as-is — caller is responsible for any prefixes.
    Otherwise uses the singleton loaded via load_model(), with the query prefix from config.
    """
    if model is not None:
        return model.encode(query, show_progress_bar=False)
    return _get_model(config).encode(f"{config.query_prefix}{query}", show_progress_bar=False)
