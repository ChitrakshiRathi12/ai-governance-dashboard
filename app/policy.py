"""
policy.py
Policy violation detection for the AI Governance Dashboard.
Combines keyword-based rules with ML classification for robust detection.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from transformers import pipeline


# ── Severity levels ───────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Violation dataclass ───────────────────────────────────────────────────────

@dataclass
class Violation:
    category:    str
    severity:    Severity
    description: str
    matched_on:  str        # keyword or classifier label that triggered it
    confidence:  float = 1.0


@dataclass
class PolicyResult:
    text:            str
    violations:      list[Violation] = field(default_factory=list)
    is_compliant:    bool = True
    overall_severity: Severity = Severity.LOW
    summary:         str = ""

    def to_dict(self) -> dict:
        return {
            "is_compliant":       self.is_compliant,
            "overall_severity":   self.overall_severity.value,
            "violation_count":    len(self.violations),
            "violations": [
                {
                    "category":    v.category,
                    "severity":    v.severity.value,
                    "description": v.description,
                    "matched_on":  v.matched_on,
                    "confidence":  v.confidence,
                }
                for v in self.violations
            ],
            "summary": self.summary,
        }


# ── Keyword rule definitions ──────────────────────────────────────────────────

POLICY_RULES = [
    {
        "category": "SELF_HARM",
        "severity": Severity.CRITICAL,
        "description": "Content that may encourage or describe self-harm or suicide",
        "keywords": [
            "kill myself", "end my life", "want to die", "commit suicide",
            "hurt myself", "self harm", "cut myself", "overdose",
        ],
    },
    {
        "category": "VIOLENCE",
        "severity": Severity.HIGH,
        "description": "Content describing or inciting physical violence",
        "keywords": [
            "beat you up", "kill you", "stab", "shoot you", "blow up",
            "attack", "murder", "assault", "strangle", "hurt you",
        ],
    },
    {
        "category": "HATE_SPEECH",
        "severity": Severity.HIGH,
        "description": "Content targeting individuals based on protected characteristics",
        "keywords": [
            "all [a-z]+ are", "those people are", "they are all",
            "inferior race", "subhuman", "go back to your country",
        ],
        "use_regex": True,
    },
    {
        "category": "MISINFORMATION",
        "severity": Severity.MEDIUM,
        "description": "Content that may spread false or unverified medical/scientific claims",
        "keywords": [
            "vaccines cause", "5g causes", "cure for cancer found",
            "doctors don't want you to know", "they are hiding the truth",
            "covid is fake", "chemtrails",
        ],
    },
    {
        "category": "PII_EXPOSURE",
        "severity": Severity.HIGH,
        "description": "Content containing or requesting personally identifiable information",
        "keywords": [],
        "patterns": [
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",          # SSN
            r"\b\d{16}\b",                                    # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",           # Phone
        ],
    },
    {
        "category": "PROMPT_INJECTION",
        "severity": Severity.CRITICAL,
        "description": "Attempted prompt injection or jailbreak of AI system",
        "keywords": [
            "ignore previous instructions", "ignore all instructions",
            "disregard your", "forget your guidelines", "you are now",
            "act as if you have no restrictions", "jailbreak",
            "pretend you are", "bypass your", "override your",
            "system prompt", "ignore your training",
        ],
    },
    {
        "category": "SEXUAL_CONTENT",
        "severity": Severity.HIGH,
        "description": "Explicit or inappropriate sexual content",
        "keywords": [
            "explicit sexual", "porn", "nude", "naked",
            "sexual act", "genitals",
        ],
    },
    {
        "category": "ILLEGAL_ACTIVITY",
        "severity": Severity.CRITICAL,
        "description": "Content describing or facilitating illegal activities",
        "keywords": [
            "how to make a bomb", "how to hack", "buy drugs",
            "launder money", "hotwire a car", "make explosives",
            "synthesize drugs", "dark web", "illegal weapons",
        ],
    },
]


# ── Classifier (loaded once) ──────────────────────────────────────────────────

_policy_classifier = None

CLASSIFIER_LABELS = [
    "hate speech",
    "violence or threats",
    "misinformation",
    "prompt injection",
    "safe content",
]


def _get_classifier():
    global _policy_classifier
    if _policy_classifier is None:
        print("Loading policy classifier...")
        _policy_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _policy_classifier


# ── Core detection functions ──────────────────────────────────────────────────

def _check_keywords(text: str) -> list[Violation]:
    """Scan text against all keyword rules."""
    violations = []
    text_lower = text.lower()

    for rule in POLICY_RULES:
        # Pattern-based check (e.g. PII regex)
        if "patterns" in rule:
            for pattern in rule["patterns"]:
                if re.search(pattern, text):
                    violations.append(Violation(
                        category=rule["category"],
                        severity=rule["severity"],
                        description=rule["description"],
                        matched_on=f"pattern: {pattern}",
                        confidence=1.0,
                    ))
                    break
            continue

        # Regex keyword check
        if rule.get("use_regex"):
            for kw in rule["keywords"]:
                if re.search(kw, text_lower):
                    violations.append(Violation(
                        category=rule["category"],
                        severity=rule["severity"],
                        description=rule["description"],
                        matched_on=kw,
                        confidence=1.0,
                    ))
                    break
            continue

        # Plain keyword check
        for kw in rule["keywords"]:
            if kw in text_lower:
                violations.append(Violation(
                    category=rule["category"],
                    severity=rule["severity"],
                    description=rule["description"],
                    matched_on=kw,
                    confidence=1.0,
                ))
                break

    return violations


def _check_classifier(text: str, threshold: float = 0.6) -> list[Violation]:
    """Run zero-shot classifier and add violations for high-confidence hits."""
    clf = _get_classifier()
    result = clf(text, CLASSIFIER_LABELS)
    scores = dict(zip(result["labels"], result["scores"]))

    violations = []

    label_map = {
        "hate speech":          ("HATE_SPEECH",       Severity.HIGH),
        "violence or threats":  ("VIOLENCE",          Severity.HIGH),
        "misinformation":       ("MISINFORMATION",    Severity.MEDIUM),
        "prompt injection":     ("PROMPT_INJECTION",  Severity.CRITICAL),
    }

    for label, (category, severity) in label_map.items():
        score = scores.get(label, 0.0)
        if score >= threshold:
            violations.append(Violation(
                category=category,
                severity=severity,
                description=f"Classifier detected '{label}' with high confidence",
                matched_on=f"classifier: {label}",
                confidence=round(score, 4),
            ))

    return violations


def _deduplicate(violations: list[Violation]) -> list[Violation]:
    """Remove duplicate violations for the same category, keeping highest confidence."""
    seen = {}
    for v in violations:
        if v.category not in seen or v.confidence > seen[v.category].confidence:
            seen[v.category] = v
    return list(seen.values())


def _overall_severity(violations: list[Violation]) -> Severity:
    if not violations:
        return Severity.LOW
    order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    for level in order:
        if any(v.severity == level for v in violations):
            return level
    return Severity.LOW


def _build_summary(violations: list[Violation]) -> str:
    if not violations:
        return "No policy violations detected. Content is compliant."
    categories = [v.category for v in violations]
    return (
        f"Detected {len(violations)} policy violation(s): "
        f"{', '.join(categories)}. "
        f"Overall severity: {_overall_severity(violations).value}."
    )


# ── Main check function ───────────────────────────────────────────────────────

def check_policy(
    text: str,
    use_classifier: bool = True,
    classifier_threshold: float = 0.6,
) -> PolicyResult:
    """
    Run full policy check on input text.

    Args:
        text: Text to evaluate.
        use_classifier: Whether to run the zero-shot classifier
                        (slower but catches more nuanced violations).
        classifier_threshold: Minimum confidence to flag a classifier result.

    Returns:
        PolicyResult with all violations, severity, and summary.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    # Keyword-based detection (fast)
    kw_violations = _check_keywords(text)

    # Classifier-based detection (slower, more nuanced)
    clf_violations = []
    if use_classifier:
        clf_violations = _check_classifier(text, threshold=classifier_threshold)

    # Combine and deduplicate
    all_violations = _deduplicate(kw_violations + clf_violations)
    severity = _overall_severity(all_violations)
    summary  = _build_summary(all_violations)

    return PolicyResult(
        text=text,
        violations=all_violations,
        is_compliant=len(all_violations) == 0,
        overall_severity=severity,
        summary=summary,
    )


def check_policy_batch(texts: list[str], use_classifier: bool = True) -> list[dict]:
    """Run policy check on a list of texts. Returns list of result dicts."""
    results = []
    for i, text in enumerate(texts):
        print(f"Checking policy for text {i + 1}/{len(texts)}...")
        try:
            result = check_policy(text, use_classifier=use_classifier)
            results.append({"text": text, **result.to_dict()})
        except Exception as e:
            results.append({"text": text, "error": str(e)})
    return results