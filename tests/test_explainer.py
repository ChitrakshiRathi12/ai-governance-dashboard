"""
test_explainer.py
Unit tests for the LIME and SHAP explainability module.
Run with: pytest tests/ -v
"""

import os
import pytest
from app.explainer import (
    explain_with_lime,
    explain_with_shap,
    plot_lime_explanation,
    plot_shap_explanation,
    plot_token_highlight,
    explain_full,
)

CLEAN_TEXT = "The weather today is sunny and pleasant."
TOXIC_TEXT = "You are completely useless and nobody wants you around."
OUTPUT_DIR = "reports/test_output"


# ── LIME tests ────────────────────────────────────────────────────────────────

class TestLIME:

    def test_returns_required_keys(self):
        result = explain_with_lime(CLEAN_TEXT, num_samples=100)
        assert set(result.keys()) >= {"method", "text", "target", "word_scores", "top_word"}

    def test_method_is_lime(self):
        result = explain_with_lime(CLEAN_TEXT, num_samples=100)
        assert result["method"] == "LIME"

    def test_word_scores_is_list(self):
        result = explain_with_lime(CLEAN_TEXT, num_samples=100)
        assert isinstance(result["word_scores"], list)

    def test_word_scores_have_correct_keys(self):
        result = explain_with_lime(TOXIC_TEXT, num_samples=100)
        for ws in result["word_scores"]:
            assert "word" in ws and "score" in ws

    def test_scores_are_floats(self):
        result = explain_with_lime(TOXIC_TEXT, num_samples=100)
        for ws in result["word_scores"]:
            assert isinstance(ws["score"], float)

    def test_empty_text_raises_error(self):
        with pytest.raises(ValueError):
            explain_with_lime("")

    def test_top_word_is_string_or_none(self):
        result = explain_with_lime(CLEAN_TEXT, num_samples=100)
        assert result["top_word"] is None or isinstance(result["top_word"], str)


# ── SHAP tests ────────────────────────────────────────────────────────────────

class TestSHAP:

    def test_returns_required_keys(self):
        result = explain_with_shap(CLEAN_TEXT, max_evals=100)
        assert set(result.keys()) >= {
            "method", "text", "target",
            "token_scores", "top_tokens", "base_value"
        }

    def test_method_is_shap(self):
        result = explain_with_shap(CLEAN_TEXT, max_evals=100)
        assert result["method"] == "SHAP"

    def test_token_scores_is_list(self):
        result = explain_with_shap(TOXIC_TEXT, max_evals=100)
        assert isinstance(result["token_scores"], list)

    def test_token_scores_have_correct_keys(self):
        result = explain_with_shap(TOXIC_TEXT, max_evals=100)
        for ts in result["token_scores"]:
            assert "token" in ts and "shap_value" in ts

    def test_base_value_is_float(self):
        result = explain_with_shap(CLEAN_TEXT, max_evals=100)
        assert isinstance(result["base_value"], float)

    def test_top_tokens_max_10(self):
        result = explain_with_shap(TOXIC_TEXT, max_evals=100)
        assert len(result["top_tokens"]) <= 10

    def test_empty_text_raises_error(self):
        with pytest.raises(ValueError):
            explain_with_shap("")


# ── Plot generation tests ─────────────────────────────────────────────────────

class TestPlots:

    def test_lime_plot_creates_file(self):
        lime_result = explain_with_lime(TOXIC_TEXT, num_samples=100)
        path = plot_lime_explanation(lime_result, output_dir=OUTPUT_DIR, filename="test_lime.png")
        assert path is not None
        assert os.path.exists(path)

    def test_shap_bar_plot_creates_file(self):
        shap_result = explain_with_shap(TOXIC_TEXT, max_evals=100)
        path = plot_shap_explanation(shap_result, output_dir=OUTPUT_DIR, filename="test_shap.png")
        assert path is not None
        assert os.path.exists(path)

    def test_token_highlight_plot_creates_file(self):
        shap_result = explain_with_shap(TOXIC_TEXT, max_evals=100)
        path = plot_token_highlight(shap_result, output_dir=OUTPUT_DIR, filename="test_highlight.png")
        assert path is not None
        assert os.path.exists(path)


# ── Full pipeline test ────────────────────────────────────────────────────────

class TestExplainFull:

    def test_returns_plots_dict(self):
        result = explain_full(
            TOXIC_TEXT,
            output_dir=OUTPUT_DIR,
            run_shap=True,
            run_lime=True,
        )
        assert "plots" in result
        assert "lime" in result
        assert "shap" in result

    def test_lime_only_mode(self):
        result = explain_full(
            CLEAN_TEXT,
            output_dir=OUTPUT_DIR,
            run_shap=False,
            run_lime=True,
        )
        assert result["lime"] != {}
        assert result["shap"] == {}

    def test_shap_only_mode(self):
        result = explain_full(
            CLEAN_TEXT,
            output_dir=OUTPUT_DIR,
            run_shap=True,
            run_lime=False,
        )
        assert result["shap"] != {}
        assert result["lime"] == {}
