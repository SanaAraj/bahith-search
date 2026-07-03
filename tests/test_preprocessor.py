from preprocessor import (
    clean_whitespace,
    normalize_arabic,
    preprocess,
    remove_diacritics,
)


def test_remove_diacritics_strips_tashkeel():
    assert remove_diacritics("مُحَمَّد") == "محمد"
    assert remove_diacritics("السَّلَامُ عَلَيْكُم") == "السلام عليكم"


def test_normalize_alef_variants_collapse_to_bare_alef():
    for variant in ("إسلام", "أسلام", "آسلام", "اسلام"):
        assert normalize_arabic(variant).startswith("ا")


def test_normalize_taa_marbuta_and_alef_maksura():
    assert normalize_arabic("مدرسة") == "مدرسه"
    assert normalize_arabic("مصطفى") == "مصطفي"


def test_clean_whitespace_collapses_runs_and_trims():
    assert clean_whitespace("  البرمجة   والتقنية  ") == "البرمجة والتقنية"


def test_preprocess_is_idempotent():
    once = preprocess("الإسلامُ  والإيمان")
    assert preprocess(once) == once


def test_preprocess_makes_diacritic_variants_match():
    # A query with diacritics should normalise to the same form as without.
    assert preprocess("عَلِيّ") == preprocess("علي")


def test_preprocess_handles_empty_string():
    assert preprocess("") == ""
