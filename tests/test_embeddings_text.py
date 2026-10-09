from __future__ import annotations

import numpy as np

from saferag.embeddings import HashingEmbedder, query_coverage
from saferag.text import normalize, numbers_in, stem, tokens


def test_tokens_keep_identifiers_and_normalise_numbers() -> None:
    toks = tokens("Use form EXP-14 for $2,000; full-time staff get 0.65/mile.")
    assert "exp-14" in toks
    assert "2000" in toks
    assert "0.65" in toks
    assert "full" in toks and "time" in toks


def test_normalize_strips_invisible_characters() -> None:
    assert normalize("ig\u200bnore\u202e  this") == "ignore this"


def test_stemming_is_consistent() -> None:
    assert stem("expense") == stem("expenses")
    assert stem("approve") == stem("approves") == stem("approval")


def test_numbers_in() -> None:
    assert numbers_in("$75 and $1,000 on EXP-14") == {"75", "1000", "exp-14"}


def test_embeddings_are_deterministic_and_normalised() -> None:
    emb = HashingEmbedder()
    a1, a2 = emb.embed(["meal limit"])[0], emb.embed(["meal limit"])[0]
    assert np.array_equal(a1, a2)
    assert abs(float(np.linalg.norm(a1)) - 1.0) < 1e-5
    assert emb.model_id == "hashing-v1-d384"


def test_concepts_bring_related_wording_closer() -> None:
    emb = HashingEmbedder()
    vacation, pto, password = emb.embed(
        ["vacation days", "paid time off PTO days", "password length rules"]
    )
    assert float(vacation @ pto) > float(vacation @ password)


def test_query_coverage_counts_concepts_once() -> None:
    assert query_coverage("vacation days", "PTO days per year") == 1.0
    assert query_coverage("parental leave", "bereavement leave days") == 0.5
