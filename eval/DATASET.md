# Retrieval evaluation dataset

`dataset.json` is a fixed set of Arabic queries, each annotated with the
document `source` file(s) judged relevant. Relevance is defined at the
**document** level: a document counts as retrieved if any of its chunks
appears in the ranked results. `run_eval.py` reports mean recall@k and MRR.

## Composition (current starter)

- 28 queries over the 12-document bundled offline corpus.
- 24 Modern Standard Arabic (`"variety": "msa"`) + 4 dialectal
  (`"variety": "dialect"`, Gulf and Egyptian) to probe dialect→MSA matching.
- Mix of topical ("ما هو الذكاء الاصطناعي"), factual ("من فاز بكأس العالم خمس
  مرات"), and paraphrase queries whose wording differs from the source text.

## Known limitation — the eval is currently saturated

On the 12-document corpus this set scores **recall@5 = 1.00, MRR = 1.00**.
Those numbers are real and reproducible, but the task is too easy to be
discriminative: each query maps to a single, topically-unique document, so a
working system trivially returns it at rank 1. The suite therefore acts as a
**regression floor** (it will fail if retrieval breaks badly) rather than a
sensitive quality signal.

To make it discriminative — an open design task, owned by Sana:

- Grow the corpus to ~30–50 documents, including **confusable / near-topic**
  pairs (e.g. space vs. astronomy, physics vs. astronomy) so ranking actually
  has to discriminate.
- Add multi-relevant queries (a question answered by 2–3 documents) and
  harder negatives.
- Add more dialectal coverage (Levantine, Maghrebi) and mark it.

Until then, treat the headline numbers as "retrieval is wired correctly," not
as a benchmark of retrieval quality.

## Regenerating

```bash
python build_index.py --offline          # deterministic corpus + indices
python -m eval.run_eval --k 5 --save      # writes eval/results/<timestamp>.json
```

Retrieval is deterministic, so runs reproduce exactly for a given embedding
model. Reports record the embedding model id and a UTC timestamp.
