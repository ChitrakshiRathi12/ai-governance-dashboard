"""
explainer.py
Explainability module for the AI Governance Dashboard.
Uses LIME and SHAP to explain which words/phrases drove toxicity
and policy violation scores for each LLM output.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lime.lime_text import LimeTextExplainer
from detoxify import Detoxify
import shap
import os
from datetime import datetime


# ── Model wrapper (LIME needs a callable) ─────────────────────────────────────

_detoxify_model = None


def _get_model():
    global _detoxify_model
    if _detoxify_model is None:
        _detoxify_model = Detoxify("original")
    return _detoxify_model


def _predict_toxicity_proba(texts: list[str]) -> np.ndarray:
    """
    Returns a 2-column probability array for LIME:
    column 0 = P(not toxic), column 1 = P(toxic)
    """
    model = _get_model()
    scores = [model.predict(t)["toxicity"] for t in texts]
    scores = np.array(scores)
    return np.column_stack([1 - scores, scores])


# ── LIME explainer ────────────────────────────────────────────────────────────

_lime_explainer = LimeTextExplainer(class_names=["not toxic", "toxic"])


def explain_with_lime(
    text: str,
    num_features: int = 10,
    num_samples: int = 300,
) -> dict:
    """
    Generates a LIME explanation for the toxicity score of the given text.

    Args:
        text:         The text to explain.
        num_features: Number of top features (words) to highlight.
        num_samples:  Number of perturbed samples LIME uses internally.

    Returns:
        A dict with word-level importance scores and metadata.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    exp = _lime_explainer.explain_instance(
        text,
        _predict_toxicity_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=[1],   # explain the "toxic" class
    )

    word_scores = exp.as_list(label=1)

    return {
        "method":       "LIME",
        "text":         text,
        "target":       "toxicity",
        "word_scores":  [{"word": w, "score": round(s, 4)} for w, s in word_scores],
        "top_word":     max(word_scores, key=lambda x: abs(x[1]))[0] if word_scores else None,
        "explanation":  exp,   # raw LIME object (for plot generation)
    }


# ── SHAP explainer ────────────────────────────────────────────────────────────

def explain_with_shap(
    text: str,
    max_evals: int = 200,
) -> dict:
    """
    Generates a SHAP explanation using the Partition explainer on token level.

    Args:
        text:      The text to explain.
        max_evals: Max evaluations for the SHAP explainer.

    Returns:
        A dict with token-level SHAP values and metadata.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    def model_fn(texts):
        model = _get_model()
        return np.array([model.predict(t)["toxicity"] for t in texts])

    masker   = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(model_fn, masker)
    shap_values = explainer([text], max_evals=max_evals, silent=True)

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    token_scores = [
        {"token": str(t), "shap_value": round(float(v), 4)}
        for t, v in zip(tokens, values)
    ]
    token_scores_sorted = sorted(token_scores, key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "method":               "SHAP",
        "text":                 text,
        "target":               "toxicity",
        "token_scores":         token_scores,
        "top_tokens":           token_scores_sorted[:10],
        "base_value":           round(float(shap_values.base_values[0]), 4),
        "shap_values_object":   shap_values,   # raw object for waterfall plot
    }


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def plot_lime_explanation(
    lime_result: dict,
    output_dir: str = "reports",
    filename: str = None,
) -> str:
    """
    Saves a horizontal bar chart of LIME word importance scores.
    Returns the file path of the saved image.
    """
    _ensure_output_dir(output_dir)
    filename = filename or f"lime_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)

    word_scores = lime_result["word_scores"]
    if not word_scores:
        return None

    words  = [ws["word"]  for ws in word_scores]
    scores = [ws["score"] for ws in word_scores]

    colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.5)))
    bars = ax.barh(words, scores, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Importance Score (positive = toxic contribution)")
    ax.set_title(f"LIME Explanation — Toxicity\n\"{lime_result['text'][:60]}...\"" if len(lime_result['text']) > 60 else f"LIME Explanation — Toxicity\n\"{lime_result['text']}\"")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def plot_shap_explanation(
    shap_result: dict,
    output_dir: str = "reports",
    filename: str = None,
) -> str:
    """
    Saves a horizontal bar chart of top SHAP token contributions.
    Returns the file path of the saved image.
    """
    _ensure_output_dir(output_dir)
    filename = filename or f"shap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)

    top_tokens = shap_result["top_tokens"]
    if not top_tokens:
        return None

    tokens = [t["token"]      for t in top_tokens]
    values = [t["shap_value"] for t in top_tokens]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tokens) * 0.5)))
    ax.barh(tokens, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"SHAP Value (base = {shap_result['base_value']:.4f})")
    ax.set_title(f"SHAP Explanation — Top Token Contributions\n\"{shap_result['text'][:60]}\"")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def plot_token_highlight(
    shap_result: dict,
    output_dir: str = "reports",
    filename: str = None,
) -> str:
    """
    Saves a colour-highlighted text image where each token is
    coloured by its SHAP value — red = toxic, green = safe.
    Returns the file path of the saved image.
    """
    _ensure_output_dir(output_dir)
    filename = filename or f"highlight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)

    token_scores = shap_result["token_scores"]
    if not token_scores:
        return None

    tokens = [t["token"]      for t in token_scores]
    values = [t["shap_value"] for t in token_scores]

    # Normalise values to [0, 1] for colourmap
    max_abs = max(abs(v) for v in values) or 1.0
    norm_values = [v / max_abs for v in values]

    cmap = plt.cm.RdYlGn_r   # red = high SHAP, green = low

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.8), 2))
    ax.axis("off")

    x = 0.02
    y = 0.5
    for token, norm_val in zip(tokens, norm_values):
        colour = cmap((norm_val + 1) / 2)
        ax.text(
            x, y, token + " ",
            ha="left", va="center",
            fontsize=13, fontweight="bold",
            color="white",
            bbox=dict(facecolor=colour, alpha=0.85, pad=3, boxstyle="round,pad=0.3"),
            transform=ax.transAxes,
        )
        x += len(token) * 0.018 + 0.02
        if x > 0.9:
            x = 0.02
            y -= 0.35

    ax.set_title("Token-level SHAP Highlights  |  🔴 Toxic  🟢 Safe", pad=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


# ── Full explanation pipeline ─────────────────────────────────────────────────

def explain_full(
    text: str,
    output_dir: str = "reports",
    run_shap: bool = True,
    run_lime: bool = True,
) -> dict:
    """
    Runs both LIME and SHAP on the text and saves all plots.

    Returns a combined dict with scores, top features, and plot paths.
    """
    result = {"text": text, "plots": {}, "lime": {}, "shap": {}}

    if run_lime:
        print("Running LIME explanation...")
        lime_result = explain_with_lime(text)
        result["lime"] = {
            "word_scores": lime_result["word_scores"],
            "top_word":    lime_result["top_word"],
        }
        lime_plot = plot_lime_explanation(lime_result, output_dir=output_dir)
        result["plots"]["lime_bar"]  = lime_plot

    if run_shap:
        print("Running SHAP explanation...")
        shap_result = explain_with_shap(text)
        result["shap"] = {
            "top_tokens":  shap_result["top_tokens"],
            "base_value":  shap_result["base_value"],
        }
        shap_plot      = plot_shap_explanation(shap_result, output_dir=output_dir)
        highlight_plot = plot_token_highlight(shap_result,  output_dir=output_dir)
        result["plots"]["shap_bar"]       = shap_plot
        result["plots"]["token_highlight"] = highlight_plot

    return result
