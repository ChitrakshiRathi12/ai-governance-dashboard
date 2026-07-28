"""
reporter.py
HTML and PDF report generator for the AI Governance Dashboard.
Combines analysis, policy, and explainability results into
a professional downloadable report.
"""

import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader


# ── Template setup ────────────────────────────────────────────────────────────

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_jinja_env   = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ── Context builder ───────────────────────────────────────────────────────────

def build_report_context(
    analysis_result: dict,
    policy_result,           # PolicyResult object
    lime_result:  dict = None,
    shap_result:  dict = None,
    lime_plot_path:      str = None,
    shap_plot_path:      str = None,
    highlight_plot_path: str = None,
) -> dict:
    """
    Assembles all analysis outputs into a single context dict
    ready for the Jinja2 template.
    """
    is_safe = (
        policy_result.is_compliant
        and analysis_result["risk_level"] == "LOW"
    )

    return {
        # Metadata
        "timestamp":         datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "word_count":        analysis_result.get("word_count", 0),

        # Input
        "text":              analysis_result["text"],

        # Summary
        "is_safe":           is_safe,
        "risk_level":        analysis_result["risk_level"],
        "violation_count":   len(policy_result.violations),

        # Toxicity
        "toxicity_score":    analysis_result["toxicity"]["toxicity"],
        "toxicity_scores":   analysis_result["toxicity"],

        # Sentiment
        "sentiment_compound": analysis_result["sentiment"]["compound"],

        # Flags
        "flags":             analysis_result["flags"],

        # Bias
        "bias_scores":       analysis_result.get("bias", {}),

        # Policy
        "violations":        policy_result.violations,

        # LIME
        "lime_word_scores":  lime_result["word_scores"]  if lime_result  else None,
        "lime_plot_path":    lime_plot_path,

        # SHAP
        "shap_top_tokens":   shap_result["top_tokens"]   if shap_result  else None,
        "shap_base_value":   shap_result["base_value"]   if shap_result  else None,
        "shap_plot_path":    shap_plot_path,
        "highlight_plot_path": highlight_plot_path,
    }


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_html_report(
    context:    dict,
    output_dir: str = "reports",
    filename:   str = None,
) -> str:
    """
    Renders the Jinja2 template with the given context and saves as HTML.

    Returns the path to the saved HTML file.
    """
    _ensure_dir(output_dir)
    filename  = filename or f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    filepath  = os.path.join(output_dir, filename)

    template  = _jinja_env.get_template("report_template.html")
    html      = template.render(**context)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report saved: {filepath}")
    return filepath


# ── PDF report ────────────────────────────────────────────────────────────────

def generate_pdf_report(
    context:    dict,
    output_dir: str = "reports",
    filename:   str = None,
) -> str:
    """
    Renders the report as a PDF using WeasyPrint.
    Falls back gracefully if WeasyPrint is not installed.

    Returns the path to the saved PDF file, or None on failure.
    """
    try:
        from weasyprint import HTML as WP_HTML
    except ImportError:
        print("WeasyPrint not installed. Run: pip install weasyprint")
        print("Falling back to HTML report only.")
        return generate_html_report(context, output_dir=output_dir)

    _ensure_dir(output_dir)
    filename  = filename or f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath  = os.path.join(output_dir, filename)

    template  = _jinja_env.get_template("report_template.html")
    html_str  = template.render(**context)

    WP_HTML(string=html_str, base_url=os.path.abspath(output_dir)).write_pdf(filepath)

    print(f"PDF report saved: {filepath}")
    return filepath


# ── Full report pipeline ──────────────────────────────────────────────────────

def generate_full_report(
    analysis_result: dict,
    policy_result,
    lime_result:         dict = None,
    shap_result:         dict = None,
    lime_plot_path:      str  = None,
    shap_plot_path:      str  = None,
    highlight_plot_path: str  = None,
    output_dir:          str  = "reports",
    export_pdf:          bool = False,
) -> dict:
    """
    Full pipeline: builds context and generates HTML (and optionally PDF).

    Returns a dict with paths to all generated files.
    """
    context = build_report_context(
        analysis_result      = analysis_result,
        policy_result        = policy_result,
        lime_result          = lime_result,
        shap_result          = shap_result,
        lime_plot_path       = lime_plot_path,
        shap_plot_path       = shap_plot_path,
        highlight_plot_path  = highlight_plot_path,
    )

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outputs  = {}

    # Always generate HTML
    html_path = generate_html_report(
        context,
        output_dir = output_dir,
        filename   = f"report_{ts}.html",
    )
    outputs["html"] = html_path

    # Optionally generate PDF
    if export_pdf:
        pdf_path = generate_pdf_report(
            context,
            output_dir = output_dir,
            filename   = f"report_{ts}.pdf",
        )
        outputs["pdf"] = pdf_path

    return outputs
