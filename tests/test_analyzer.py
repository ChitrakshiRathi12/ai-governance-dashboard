"""
test_analyzer.py
Unit tests for the core analysis engine.
Run with: pytest tests/ -v
"""

import pytest
from app.analyzer import (
    score_toxicity,
    score_sentiment,
    analyze,
    analyze_batch,
    results_to_dataframe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

CLEAN_TEXT   = "The weather today is sunny and pleasant."
TOXIC_TEXT   = "I hate you and you should disappear forever."
NEGATIVE_TEXT = "This is terrible, awful, and completely wrong."


# ── Toxicity tests ────────────────────────────────────────────────────────────

class TestToxicityScorer:

    def test_returns_all_keys(self):
        result = score_toxicity(CLEAN_TEXT)
        expected_keys = {
            "toxicity", "severe_toxicity", "obscene",
            "threat", "insult", "identity_attack"
        }
        assert set(result.keys()) == expected_keys

    def test_scores_are_floats_between_0_and_1(self):
        result = score_toxicity(CLEAN_TEXT)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} score {val} out of range"

    def test_clean_text_has_low_toxicity(self):
        result = score_toxicity(CLEAN_TEXT)
        assert result["toxicity"] < 0.3

    def test_toxic_text_has_higher_toxicity(self):
        clean  = score_toxicity(CLEAN_TEXT)["toxicity"]
        toxic  = score_toxicity(TOXIC_TEXT)["toxicity"]
        assert toxic > clean


# ── Sentiment tests ───────────────────────────────────────────────────────────

class TestSentimentScorer:

    def test_returns_all_keys(self):
        result = score_sentiment(CLEAN_TEXT)
        assert set(result.keys()) == {"positive", "negative", "neutral", "compound"}

    def test_compound_range(self):
        result = score_sentiment(CLEAN_TEXT)
        assert -1.0 <= result["compound"] <= 1.0

    def test_negative_text_has_negative_compound(self):
        result = score_sentiment(NEGATIVE_TEXT)
        assert result["compound"] < 0

    def test_clean_text_has_non_negative_compound(self):
        result = score_sentiment(CLEAN_TEXT)
        assert result["compound"] >= 0


# ── Full pipeline tests ───────────────────────────────────────────────────────

class TestAnalyzePipeline:

    def test_returns_required_keys(self):
        result = analyze(CLEAN_TEXT, include_bias=False)
        required = {"text", "timestamp", "toxicity", "sentiment",
                    "bias", "flags", "risk_level", "word_count"}
        assert required.issubset(set(result.keys()))

    def test_risk_level_values(self):
        result = analyze(CLEAN_TEXT, include_bias=False)
        assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_flags_is_list(self):
        result = analyze(CLEAN_TEXT, include_bias=False)
        assert isinstance(result["flags"], list)

    def test_word_count_correct(self):
        text = "one two three four five"
        result = analyze(text, include_bias=False)
        assert result["word_count"] == 5

    def test_empty_text_raises_error(self):
        with pytest.raises(ValueError):
            analyze("", include_bias=False)

    def test_whitespace_only_raises_error(self):
        with pytest.raises(ValueError):
            analyze("   ", include_bias=False)


# ── Batch analysis tests ──────────────────────────────────────────────────────

class TestBatchAnalysis:

    def test_returns_correct_count(self):
        texts = [CLEAN_TEXT, NEGATIVE_TEXT]
        results = analyze_batch(texts, include_bias=False)
        assert len(results) == 2

    def test_dataframe_has_rows(self):
        texts = [CLEAN_TEXT, NEGATIVE_TEXT]
        results = analyze_batch(texts, include_bias=False)
        df = results_to_dataframe(results)
        assert len(df) == 2

    def test_dataframe_has_toxicity_column(self):
        texts = [CLEAN_TEXT]
        results = analyze_batch(texts, include_bias=False)
        df = results_to_dataframe(results)
        assert "toxicity" in df.columns
