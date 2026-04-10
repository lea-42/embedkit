import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import embedkit.embedder as vmod
import numpy as np
import pytest

from embedkit.embedder import (
    DEFAULT_MODEL,
    E5_MULTILINGUAL_BASE,
    E5_MULTILINGUAL_SMALL,
    MINILM_MULTILINGUAL,
    SUPPORTED_MODELS,
    _chunk_to_text,
    _get_model,
    embed_chunks,
    embed_query,
)


CHUNKS = [
    {"chunk_number": 0, "page_number": 0, "section_breadcrumbs": ["Intro"], "text": "First chunk text."},
    {"chunk_number": 1, "page_number": 0, "section_breadcrumbs": ["Intro"], "text": "Second chunk text."},
    {"chunk_number": 2, "page_number": 1, "section_breadcrumbs": ["Benefits"], "text": "Third chunk text."},
]

FAKE_EMBEDDINGS = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
FAKE_QUERY_EMBEDDING = np.array([0.1, 0.2, 0.3], dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    vmod._model = None
    vmod._model_config = None
    yield
    vmod._model = None
    vmod._model_config = None


@pytest.fixture
def loaded_model():
    """Inject a fake loaded model into the singleton."""
    mock = MagicMock()
    mock.encode.return_value = FAKE_EMBEDDINGS
    vmod._model = mock
    vmod._model_config = DEFAULT_MODEL
    return mock


# --- model config ---

def test_default_model_is_e5_multilingual_base() -> None:
    assert DEFAULT_MODEL == E5_MULTILINGUAL_BASE


def test_supported_models_contains_all_configs() -> None:
    assert E5_MULTILINGUAL_BASE in SUPPORTED_MODELS
    assert E5_MULTILINGUAL_SMALL in SUPPORTED_MODELS
    assert MINILM_MULTILINGUAL in SUPPORTED_MODELS


def test_e5_models_have_prefixes() -> None:
    assert E5_MULTILINGUAL_BASE.query_prefix == "query: "
    assert E5_MULTILINGUAL_BASE.passage_prefix == "passage: "
    assert E5_MULTILINGUAL_SMALL.query_prefix == "query: "
    assert E5_MULTILINGUAL_SMALL.passage_prefix == "passage: "


def test_minilm_has_no_prefixes() -> None:
    assert MINILM_MULTILINGUAL.query_prefix == ""
    assert MINILM_MULTILINGUAL.passage_prefix == ""


def test_embedding_model_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        E5_MULTILINGUAL_BASE.model_name = "something-else"  # type: ignore[misc]


# --- load_model ---

def test_load_model_sets_singleton() -> None:
    mock = MagicMock()
    with patch("embedkit.embedder.SentenceTransformer", return_value=mock):
        asyncio.run(vmod.load_model(DEFAULT_MODEL))
    assert vmod._model is mock
    assert vmod._model_config == DEFAULT_MODEL


def test_load_model_loads_only_once() -> None:
    mock = MagicMock()
    with patch("embedkit.embedder.SentenceTransformer", return_value=mock) as mock_cls:
        async def run_twice() -> None:
            await vmod.load_model(DEFAULT_MODEL)
            await vmod.load_model(DEFAULT_MODEL)
        asyncio.run(run_twice())
    mock_cls.assert_called_once()


def test_load_model_invalid_s3_path_raises(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_S3_PATH", "not-a-valid-path")
    with pytest.raises(ValueError, match="MODEL_S3_PATH must be in the form"):
        asyncio.run(vmod.load_model())


def test_load_model_s3_path_missing_prefix_raises(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_S3_PATH", "s3://bucket-only")
    with pytest.raises(ValueError, match="MODEL_S3_PATH must be in the form"):
        asyncio.run(vmod.load_model())


def test_load_model_s3_path_empty_prefix_raises(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_S3_PATH", "s3://bucket/")
    with pytest.raises(ValueError, match="MODEL_S3_PATH must be in the form"):
        asyncio.run(vmod.load_model())


# --- _get_model ---

def test_get_model_raises_if_not_loaded() -> None:
    with pytest.raises(RuntimeError, match="Model not loaded"):
        _get_model(DEFAULT_MODEL)


def test_get_model_raises_if_wrong_config() -> None:
    vmod._model = MagicMock()
    vmod._model_config = E5_MULTILINGUAL_SMALL
    with pytest.raises(RuntimeError, match="Requested model"):
        _get_model(E5_MULTILINGUAL_BASE)


def test_get_model_returns_singleton(loaded_model) -> None:
    result = _get_model(DEFAULT_MODEL)
    assert result is loaded_model


# --- _chunk_to_text ---

def test_chunk_to_text_single_string_key() -> None:
    chunk = {"text": "hello world"}
    assert _chunk_to_text(chunk, ["text"]) == "hello world"


def test_chunk_to_text_list_field_joined_with_newline() -> None:
    chunk = {"section_breadcrumbs": ["Intro", "Benefits"]}
    assert _chunk_to_text(chunk, ["section_breadcrumbs"]) == "Intro\nBenefits"


def test_chunk_to_text_multiple_keys_joined_with_double_newline() -> None:
    chunk = {"section_breadcrumbs": ["Intro", "Benefits"], "text": "some text"}
    assert _chunk_to_text(chunk, ["section_breadcrumbs", "text"]) == "Intro\nBenefits\n\nsome text"


def test_chunk_to_text_single_breadcrumb() -> None:
    chunk = {"section_breadcrumbs": ["Intro"], "text": "some text"}
    assert _chunk_to_text(chunk, ["section_breadcrumbs", "text"]) == "Intro\n\nsome text"


def test_chunk_to_text_string_and_list_fields() -> None:
    chunk = {"title": "My Doc", "tags": ["a", "b"], "text": "body"}
    assert _chunk_to_text(chunk, ["title", "tags", "text"]) == "My Doc\n\na\nb\n\nbody"


# --- embed_chunks ---

def test_embed_chunks_yields_chunk_embedding_pairs(loaded_model) -> None:
    results = list(embed_chunks(CHUNKS))
    assert len(results) == 3
    chunk, embedding = results[0]
    assert chunk == CHUNKS[0]
    assert isinstance(embedding, np.ndarray)


def test_embed_chunks_pairs_match_chunks(loaded_model) -> None:
    results = list(embed_chunks(CHUNKS))
    for i, (chunk, embedding) in enumerate(results):
        assert chunk == CHUNKS[i]
        np.testing.assert_array_equal(embedding, FAKE_EMBEDDINGS[i])


def test_embed_chunks_applies_passage_prefix(loaded_model) -> None:
    list(embed_chunks(CHUNKS, config=E5_MULTILINGUAL_BASE))
    texts = loaded_model.encode.call_args[0][0]
    assert texts == [f"passage: {c['text']}" for c in CHUNKS]


def test_embed_chunks_no_prefix_for_minilm() -> None:
    mock = MagicMock()
    mock.encode.return_value = FAKE_EMBEDDINGS
    vmod._model = mock
    vmod._model_config = MINILM_MULTILINGUAL
    list(embed_chunks(CHUNKS, config=MINILM_MULTILINGUAL))
    texts = mock.encode.call_args[0][0]
    assert texts == [c["text"] for c in CHUNKS]


def test_embed_chunks_empty_input_yields_nothing(loaded_model) -> None:
    results = list(embed_chunks([]))
    assert results == []
    loaded_model.encode.assert_not_called()


def test_embed_chunks_custom_model_skips_prefix(loaded_model) -> None:
    custom = MagicMock()
    custom.encode.return_value = FAKE_EMBEDDINGS
    list(embed_chunks(CHUNKS, model=custom))
    texts = custom.encode.call_args[0][0]
    assert texts == [c["text"] for c in CHUNKS]


def test_embed_chunks_raises_if_no_singleton_and_no_model() -> None:
    with pytest.raises(RuntimeError, match="Model not loaded"):
        list(embed_chunks(CHUNKS))


def test_embed_chunks_custom_text_keys(loaded_model) -> None:
    list(embed_chunks(CHUNKS, text_keys=["section_breadcrumbs", "text"], config=E5_MULTILINGUAL_BASE))
    texts = loaded_model.encode.call_args[0][0]
    expected = [
        f"passage: {_chunk_to_text(c, ['section_breadcrumbs', 'text'])}" for c in CHUNKS
    ]
    assert texts == expected


# --- embed_query ---

def test_embed_query_returns_array(loaded_model) -> None:
    loaded_model.encode.return_value = FAKE_QUERY_EMBEDDING
    result = embed_query("what is covered?")
    assert isinstance(result, np.ndarray)


def test_embed_query_applies_query_prefix(loaded_model) -> None:
    loaded_model.encode.return_value = FAKE_QUERY_EMBEDDING
    embed_query("what is covered?", config=E5_MULTILINGUAL_BASE)
    text = loaded_model.encode.call_args[0][0]
    assert text == "query: what is covered?"


def test_embed_query_no_prefix_for_minilm() -> None:
    mock = MagicMock()
    mock.encode.return_value = FAKE_QUERY_EMBEDDING
    vmod._model = mock
    vmod._model_config = MINILM_MULTILINGUAL
    embed_query("what is covered?", config=MINILM_MULTILINGUAL)
    text = mock.encode.call_args[0][0]
    assert text == "what is covered?"


def test_embed_query_custom_model_no_prefix(loaded_model) -> None:
    custom = MagicMock()
    custom.encode.return_value = FAKE_QUERY_EMBEDDING
    embed_query("what is covered?", model=custom)
    text = custom.encode.call_args[0][0]
    assert text == "what is covered?"


def test_embed_query_raises_if_not_loaded() -> None:
    with pytest.raises(RuntimeError, match="Model not loaded"):
        embed_query("what is covered?")
