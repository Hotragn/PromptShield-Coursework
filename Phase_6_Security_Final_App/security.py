"""
security.py — QuizLab Security Module
Assignment 7: Prompt Hacking & Security

Implements four layers of defense:
  1. scan_for_injection     — detect prompt injection in any text block
  2. sanitize_topic         — validate and clean the user-supplied topic field
  3. classify_document      — heuristically detect website printouts / non-slides
  4. validate_topic_relevance — LLM check: is the topic covered in the document?
"""

import json
import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Injection detection patterns
# Each tuple: (regex_pattern, human-readable label)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (
    r"(all\s+)?ai\s+systems?\s+.{0,40}(required|must|instructed|should)\s+to\s+.{0,30}(response|reply|output)",
    "ai system instruction",
    ),
(
    r"takes?\s+precedence\s+over\s+.{0,30}(standard|normal|default|existing|previous)\s+.{0,20}(behavior|instructions?|generation|rules?)",
    "precedence override",
    ),
    (
        # Allow extra words between "previous" and "instructions" (e.g. "PREVIOUS QUIZ GENERATION INSTRUCTIONS")
        r"ignore\s+(all\s+)?(previous|prior|above|your)\s+.{0,40}(instructions?|rules?|guidelines?|constraints?)",
        "instruction override",
    ),
    (
        r"\[SYSTEM\s+OVERRIDE\]|\[OVERRIDE\]",
        "system override tag",
    ),
    (
        r"(respond|reply|answer|output|write)\s+(only\s+)?(in|using)\s+(french|spanish|german|japanese|chinese|another\s+language)",
        "language switch injection",
    ),
    (
        # [\s\S] instead of . so the match can cross a line break
        r"(output|write|begin|start)\s+[\s\S]{0,60}(phrase|text|string)\s*['\"]",
        "output redirect",
    ),
    (
        r"followed\s+by\s+[\s\S]{0,50}system\s+instructions?",
        "system instruction leak",
    ),
    (
        r"forget\s+(everything|all|your)\s*(previous|prior|above)?\s*(instructions?|training|context|role)",
        "instruction wipe",
    ),
    (
        r"(override|disregard|bypass|circumvent)\s+(your|all|the|any)\s+"
        r"(instructions?|system\s*prompt|guidelines?|rules?|restrictions?|safety)",
        "instruction bypass",
    ),
    (
        r"(you\s+are\s+now|from\s+now\s+on\s+you\s+are)\s+(a|an|the)\s+\w+\s+"
        r"(without|with\s+no)\s+(restrictions?|limits?|guidelines?|ethics?)",
        "role override",
    ),
    (
        r"(new|updated|revised|actual)\s+instructions?\s*:",
        "instruction injection",
    ),
    (
        r"\[?system\]?\s*(message|prompt|instruction)\s*:",
        "system prompt injection",
    ),
    (
        r"(\[SYSTEM\]|\[INST\]|<system>|<<SYS>>|<</SYS>>)",
        "system tag injection",
    ),
    (
        r"(reveal|output|print|show|display|expose|leak)\s+(your|the|all)\s+"
        r"(system\s+prompt|instructions?|api\s+key|credentials?|secrets?)",
        "info extraction",
    ),
    (
        r"(jailbreak|DAN\s+mode|developer\s+mode|training\s+mode|sudo\s+mode|god\s+mode|unrestricted\s+mode)",
        "jailbreak keyword",
    ),
    (
        r"(pretend|act\s+as\s+if|behave\s+as\s+if)\s+you\s+"
        r"(have\s+no|are\s+without|don'?t\s+have)\s+(restrictions?|limits?|rules?|guidelines?|ethics?)",
        "restriction bypass",
    ),
    (
        r"instead\s+of\s+(generating|creating|making|producing)\s+a?\s*(quiz|question)",
        "quiz generation bypass",
    ),
    (
        r"do\s+not\s+(generate|create|make|produce)\s+a?\s*(quiz|question)",
        "generation refusal injection",
    ),
    (
        r"(output|respond\s+with|say|write|return)\s+['\"]?"
        r"(I\s+have\s+been\s+(hacked|compromised)|pwned|owned)['\"]?",
        "injection verification phrase",
    ),
]

# Patterns suggesting the PDF is a website printout rather than lecture slides
_WEBSITE_PATTERNS: List[str] = [
    r"(home|about|contact|sign\s+in|log\s+in|sign\s+up|register|subscribe)\s*[|·•]\s*"
    r"(home|about|contact|sign\s+in|log\s+in|sign\s+up|register)",
    r"https?://\S{10,}",
    r"(cookie\s+policy|privacy\s+policy|terms\s+of\s+(service|use)|©\s*\d{4})",
    r"(follow\s+us\s+on|share\s+this|tweet\s+this|post\s+to\s+facebook)",
    r"(search\s+results?|showing\s+\d+\s+(results?|items?|products?))",
    r"(add\s+to\s+cart|buy\s+now|checkout|wishlist)",
    r"(read\s+more|load\s+more|show\s+more|view\s+all)\s*[»>→]",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_for_injection(text: str) -> Dict[str, Any]:
    """
    Scan a block of text for prompt injection patterns.

    Works on both user-supplied topic strings and extracted PDF content.

    Returns
    -------
    dict with keys:
        detected      bool — any pattern matched
        risk_level    "none" | "medium" | "high"
        patterns_found  list[str] — labels of matched patterns
        message       str — human-readable summary (empty when nothing detected)
    """
    text_lower = (text or "").lower()
    matched: List[str] = []

    for pattern, label in _INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(label)

    risk_level = "none"
    if len(matched) >= 2:
        risk_level = "high"
    elif len(matched) == 1:
        risk_level = "medium"

    return {
        "detected": bool(matched),
        "risk_level": risk_level,
        "patterns_found": matched,
        "message": _injection_message(matched, risk_level),
    }


def sanitize_topic(topic: str) -> Dict[str, Any]:
    """
    Validate and clean the topic input field.

    - Blocks high-risk topics that contain clear injection attempts.
    - Strips multi-line payloads to first line only.
    - Enforces a 200-character limit.

    Returns
    -------
    dict with keys:
        safe        bool — False means quiz generation should be blocked
        sanitized   str  — cleaned topic string (empty when safe=False)
        warning     str  — advisory message (empty when safe=True and no concerns)
        original    str  — raw input
    """
    if not topic or not topic.strip():
        return {
            "safe": False,
            "sanitized": "",
            "warning": "Topic cannot be empty.",
            "original": topic or "",
        }

    scan = scan_for_injection(topic)

    if scan["detected"] and scan["risk_level"] == "high":
        return {
            "safe": False,
            "sanitized": "",
            "warning": (
                f"Blocked: your topic contains suspicious content "
                f"({', '.join(scan['patterns_found'])}). "
                "Please enter a genuine academic subject (e.g. 'Transformers in NLP')."
            ),
            "original": topic,
        }

    # Take first line only and cap length
    cleaned = topic.strip().split("\n")[0].strip()
    if len(cleaned) > 200:
        cleaned = cleaned[:200].strip()

    return {
        "safe": True,
        "sanitized": cleaned,
        "warning": scan["message"] if scan["detected"] else "",
        "original": topic,
    }


def classify_document(text: str) -> Dict[str, Any]:
    """
    Heuristically determine whether the uploaded PDF resembles academic lecture
    slides, a website printout, or another document type.

    Returns
    -------
    dict with keys:
        document_type   "slides" | "website" | "other" | "unknown"
        confidence      "high" | "medium" | "low"
        warning         str — advisory message (empty when no concern)
    """
    text_lower = (text or "").lower()
    text_len = len(text_lower)

    if text_len < 200:
        return {
            "document_type": "unknown",
            "confidence": "high",
            "warning": (
                "The uploaded document has very little readable text. "
                "Check that the PDF is not scanned/image-only and that it contains "
                "your actual lecture content."
            ),
        }

    website_hits = sum(
        1 for p in _WEBSITE_PATTERNS
        if re.search(p, text_lower, re.IGNORECASE)
    )
    url_count = len(re.findall(r"https?://\S{10,}", text))
    nav_pattern = bool(
        re.search(
            r"(home|about|contact|menu)\s*[|·•]\s*(home|about|contact|menu)",
            text_lower,
            re.IGNORECASE,
        )
    )
    commerce_pattern = bool(
        re.search(r"(add\s+to\s+cart|checkout|buy\s+now|\$\d+\.\d{2})", text_lower)
    )

    if website_hits >= 3 or (url_count >= 5 and nav_pattern) or commerce_pattern:
        return {
            "document_type": "website",
            "confidence": "high",
            "warning": (
                "The uploaded document looks like a website printout, not lecture slides. "
                "QuizLab works best with academic lecture notes or textbook PDFs. "
                "Questions may be irrelevant or poorly grounded."
            ),
        }
    elif website_hits >= 2 or (website_hits >= 1 and url_count >= 5):
        return {
            "document_type": "other",
            "confidence": "medium",
            "warning": (
                "The uploaded document may not be standard lecture slides "
                "(possible website or article content detected). "
                "Quiz grounding quality may be reduced."
            ),
        }

    return {
        "document_type": "slides",
        "confidence": "high",
        "warning": "",
    }


def validate_topic_relevance(context: str, topic: str, llm) -> Dict[str, Any]:
    """
    Use the provided LangChain LLM to assess whether the requested topic is
    meaningfully covered in the retrieved document context.

    Parameters
    ----------
    context : str — retrieved lecture text
    topic   : str — user-supplied quiz topic
    llm     : LangChain ChatOpenAI instance

    Returns
    -------
    dict with keys:
        relevant          bool
        confidence        "high" | "medium" | "low"
        coverage_estimate int   0–100
        reason            str   one-sentence explanation
        warning           str   advisory (empty when coverage ≥ 50%)
    """
    if not context.strip() or not topic.strip():
        return {
            "relevant": False,
            "confidence": "low",
            "coverage_estimate": 0,
            "reason": "Empty context or topic.",
            "warning": (
                "Could not validate topic relevance: no document content or topic found."
            ),
        }

    prompt = (
        "You are checking whether a student's quiz topic is covered in their uploaded lecture notes.\n\n"
        f'Topic: "{topic}"\n\n'
        "Lecture notes excerpt (first 2000 characters):\n"
        "---\n"
        f"{context[:2000]}\n"
        "---\n\n"
        "Reply ONLY with valid JSON — no markdown, no extra text:\n"
        '{"relevant": true_or_false, "confidence": "high"|"medium"|"low", '
        '"coverage_estimate": 0_to_100, "reason": "one sentence"}'
    )

    try:
        response = llm.invoke(prompt)
        content = (response.content or "").strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            relevant = bool(data.get("relevant", True))
            coverage = max(0, min(100, int(data.get("coverage_estimate", 50))))
            confidence = str(data.get("confidence", "low"))
            reason = str(data.get("reason", ""))

            warning = ""
            if not relevant or coverage < 20:
                warning = (
                    f"Warning: '{topic}' does not appear to be well-covered in the uploaded "
                    f"document (estimated coverage: {coverage}%). {reason} "
                    "The generated quiz may not be grounded in your slides."
                )
            elif coverage < 50:
                warning = (
                    f"Note: Limited coverage of '{topic}' found in the uploaded document "
                    f"(estimated {coverage}%). {reason} "
                    "Some questions may be based on partial context."
                )

            return {
                "relevant": relevant,
                "confidence": confidence,
                "coverage_estimate": coverage,
                "reason": reason,
                "warning": warning,
            }
    except Exception:
        pass

    # Graceful fallback — don't block generation on validation failure
    return {
        "relevant": True,
        "confidence": "low",
        "coverage_estimate": 50,
        "reason": "Validation check could not complete.",
        "warning": "",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _injection_message(patterns: List[str], risk_level: str) -> str:
    if not patterns:
        return ""
    label_str = ", ".join(patterns)
    if risk_level == "high":
        return (
            f"High-risk content detected in document ({label_str}). "
            "This content may contain prompt injection instructions designed to "
            "hijack quiz generation. Generation has been blocked."
        )
    return (
        f"Suspicious content detected in document ({label_str}). "
        "Quiz generation will proceed, but only content verified as lecture "
        "material is used."
    )
