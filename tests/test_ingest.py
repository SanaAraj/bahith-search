import ingest
from ingest import chunk_text, process_documents


def test_chunk_text_short_returns_single_chunk():
    assert chunk_text("نص قصير") == ["نص قصير"]


def test_chunk_text_long_splits_into_multiple_chunks():
    text = "جملة عربية طويلة. " * 200  # ~3600 chars
    chunks = chunk_text(text, chunk_size=800, overlap=200)
    assert len(chunks) > 1
    # Overlap means total chunked length exceeds the source length.
    assert sum(len(c) for c in chunks) > len(text)


def test_chunk_text_drops_tiny_trailing_fragments():
    # Multi-chunk path filters fragments of <= 50 chars.
    chunks = chunk_text("ا" * 900 + "." + "ب" * 10, chunk_size=800, overlap=200)
    assert all(len(c) > 50 for c in chunks)


def test_process_documents_emits_stable_ids_and_required_keys(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "read_documents",
        lambda: [{"title": "الذكاء", "content": "محتوى قصير عن الذكاء", "source": "ai.txt"}],
    )
    docs = process_documents()
    assert docs
    first = docs[0]
    assert set(first.keys()) == {"id", "title", "content", "source", "chunk_index"}
    assert first["id"] == "ai.txt_0"
