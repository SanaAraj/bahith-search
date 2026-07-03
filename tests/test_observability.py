import observability
from observability import trace_query


def test_tracing_disabled_yields_null_trace(monkeypatch):
    # Force the disabled path regardless of ambient env.
    monkeypatch.setattr(observability, "_get_client", lambda: None)

    with trace_query("سؤال", mode="hybrid", top_k=5) as tr:
        # Null trace must accept spans and updates without raising.
        with tr.span("retrieval", mode="hybrid"):
            pass
        tr.update(output={"num_results": 0})


def test_is_enabled_false_when_no_client(monkeypatch):
    monkeypatch.setattr(observability, "_get_client", lambda: None)
    assert observability.is_enabled() is False
