#!/usr/bin/env python
"""
Multilingual Advanced Evaluation Prototype

Features:
1. Dataset: 10 base English examples, each translated into ES, DE, ZH (30 rows total).
2. MT: Uses OpenAI to translate EN -> target_lang.
3. Back-translation: target_lang -> EN, with simple Jaccard similarity.
4. LLM-as-a-judge:
   - MQM-style error categories (simplified)
   - adequacy / fluency / style_consistency / overall_score
   - fluency_comment
5. QA checks:
   - length_ratio
   - numbers preservation
   - placeholders
   - punctuation
   - CJK script check
   - HTML/Markdown/JSON-ish structure (balanced brackets, backticks)
   - entity preservation (URLs, emails)
   - untranslated key terms (English terms that remain where they likely should localize)
6. Metrics:
   - Average fluency score per language
   - Frequency of placeholder errors
   - Percentage of segments with any QA flags
   - Standard deviation of length_ratio per language
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI

# ----------------------
# Config
# ----------------------

TRANSLATION_MODEL = "gpt-4.1-mini"
EVAL_MODEL = "gpt-4.1-mini"
BACKTRANSLATION_MODEL = "gpt-4.1"

OUTPUT_CSV = "multilingual_advanced_eval.csv"
OUTPUT_JSON = "multilingual_advanced_eval.json"

client = OpenAI()  # uses OPENAI_API_KEY from env


# ----------------------
# Dataset: 10 base examples × 3 languages
# ----------------------

BASE_SENTENCES = [
    "Welcome back! We’ve added new features to help you work faster and smarter.",
    "Your security code will expire in 10 minutes. Do not share it with anyone.",
    "By clicking \"Accept\", you agree to our updated Privacy Policy and Terms of Service.",
    "This promotion is valid only in selected countries and cannot be combined with other offers.",
    "Your payment of $249.00 has been received. A receipt has been sent to your email address.",
    "Download the mobile app to receive real-time notifications and personalized recommendations.",
    "Some advanced settings may affect performance. Change them only if you know what you’re doing.",
    "We’re processing a high volume of requests. Response times may be slower than usual.",
    "For API access, generate a new key in the Developer Console and keep it secure.",
    "If you believe this activity is suspicious, please reset your password immediately."
]

TARGET_LANGS = ["es", "de", "zh"]


def build_dataset() -> pd.DataFrame:
    rows = []
    row_id = 5000
    for idx, source_text in enumerate(BASE_SENTENCES, start=1):
        for lang in TARGET_LANGS:
            rows.append(
                {
                    "id": row_id,
                    "group_id": idx,  # same group_id for the same English sentence across langs
                    "source_lang": "en",
                    "target_lang": lang,
                    "source_text": source_text,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


# ----------------------
# MT: translation & back-translation
# ----------------------

def translate_with_openai(text: str, source_lang: str, target_lang: str) -> str:
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        f"Return only the translated text, with no explanations or quotes.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        response = client.responses.create(
            model=TRANSLATION_MODEL,
            input=prompt,
        )
        content = response.output[0].content[0].text
        return content.strip()
    except Exception as e:
        print(f"[WARN] Translation failed ({e}). Using dummy output.")
        return f"[MT_FAILED_{target_lang}]: {text}"


def backtranslate_with_openai(text: str, source_lang: str) -> str:
    # Here source_lang is the language of `text` (es/de/zh), target is English
    prompt = (
        f"Translate the following text from {source_lang} to English. "
        f"Return only the translated text, with no explanations or quotes.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        response = client.responses.create(
            model=BACKTRANSLATION_MODEL,
            input=prompt,
        )
        content = response.output[0].content[0].text
        return content.strip()
    except Exception as e:
        print(f"[WARN] Back-translation failed ({e}). Using dummy output.")
        return f"[BACK_MT_FAILED_EN]: {text}"


def add_mt_and_backtranslation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mt_outputs = []
    backtranslations = []
    for _, row in df.iterrows():
        mt = translate_with_openai(
            row["source_text"], row["source_lang"], row["target_lang"]
        )
        bt = backtranslate_with_openai(mt, row["target_lang"])
        mt_outputs.append(mt)
        backtranslations.append(bt)

    df["mt_output"] = mt_outputs
    df["back_translation_en"] = backtranslations
    return df


# ----------------------
# Similarity (source vs back-translation)
# ----------------------

def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def add_backtranslation_similarity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sims = []
    for _, row in df.iterrows():
        src = str(row["source_text"])
        bt = str(row["back_translation_en"])
        sims.append(jaccard_similarity(src, bt))
    df["backtranslation_similarity"] = sims
    return df


# ----------------------
# LLM-as-a-judge (MQM-like)
# ----------------------

def evaluate_translation_mqm(
    source_text: str, mt_output: str, source_lang: str, target_lang: str
) -> Dict[str, Any]:
    """
    Ask OpenAI to evaluate translation with:
      - adequacy, fluency, style_consistency, overall_score (0-1 floats)
      - fluency_comment
      - errors: list of {category, severity, note}
    """

    system_msg = (
        "You are a multilingual localization quality expert. "
        "You evaluate translations using a simplified MQM framework."
    )

    user_msg = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        f"Source text:\n{source_text}\n\n"
        f"Translated text:\n{mt_output}\n\n"
        "Evaluate this translation and respond ONLY with a JSON object. "
        "Use the following format:\n"
        "{\n"
        "  \"adequacy\": float between 0 and 1,\n"
        "  \"fluency\": float between 0 and 1,\n"
        "  \"style_consistency\": float between 0 and 1,\n"
        "  \"overall_score\": float between 0 and 1,\n"
        "  \"fluency_comment\": \"short natural-language explanation\",\n"
        "  \"errors\": [\n"
        "    {\n"
        "      \"category\": \"accuracy|fluency|terminology|style|localization|other\",\n"
        "      \"severity\": \"minor|major|critical\",\n"
        "      \"note\": \"short description of the issue\"\n"
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "Do not include any extra text, markdown, or explanation outside the JSON."
    )

    try:
        response = client.responses.create(
            model=EVAL_MODEL,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        content = response.output[0].content[0].text

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {
                "adequacy": None,
                "fluency": None,
                "style_consistency": None,
                "overall_score": None,
                "fluency_comment": f"RAW_OUTPUT: {content}",
                "errors": [],
            }

    except Exception as e:
        print(f"[WARN] Evaluation failed ({e}). Using dummy scores.")
        data = {
            "adequacy": None,
            "fluency": None,
            "style_consistency": None,
            "overall_score": None,
            "fluency_comment": f"LLM eval not available: {e}",
            "errors": [],
        }

    # Normalize errors to a JSON-serializable list
    if not isinstance(data.get("errors"), list):
        data["errors"] = []
    return data


def add_llm_mqm_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    eval_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        eval_dict = evaluate_translation_mqm(
            row["source_text"],
            row["mt_output"],
            row["source_lang"],
            row["target_lang"],
        )
        eval_records.append(eval_dict)

    df_eval = pd.DataFrame(eval_records)
    return pd.concat([df.reset_index(drop=True), df_eval.reset_index(drop=True)], axis=1)


# ----------------------
# QA: structure, entities, placeholders, etc.
# ----------------------

PLACEHOLDER_PATTERN = r"{[^}]+}|%s|%d|%f|{{[^}]+}}"

KEY_TERMS = [
    "Privacy Policy",
    "Terms of Service",
    "API",
    "SDK",
    "JSON",
    "Developer Console",
    "Pro plan",
]

URL_PATTERN = r"https?://[^\s]+"
EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"


def extract_numbers(text: str) -> List[str]:
    if text is None:
        return []
    return re.findall(r"\d+(?:\.\d+)?", str(text))


def length_ratio(source: str, target: str) -> float:
    source = "" if source is None else str(source)
    target = "" if target is None else str(target)
    if not source:
        return 1.0
    return len(target) / len(source)


def sentence_final_punct(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return text[-1] if text and text[-1] in ".!?" else ""


def extract_placeholders(text: str):
    if text is None:
        return set()
    return set(re.findall(PLACEHOLDER_PATTERN, str(text)))


def contains_cjk(text: str) -> bool:
    if text is None:
        return False
    text = str(text)
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def balanced_brackets(text: str) -> List[str]:
    """
    Very simple checks for balanced (), {}, [], and backticks `.
    Returns a list of issues.
    """
    issues = []
    if text is None:
        return issues

    s = str(text)

    def check_pair(open_ch, close_ch, label):
        stack = 0
        for ch in s:
            if ch == open_ch:
                stack += 1
            elif ch == close_ch:
                stack -= 1
            if stack < 0:
                issues.append(f"unbalanced_{label}")
                return
        if stack != 0:
            issues.append(f"unbalanced_{label}")

    check_pair("(", ")", "parentheses")
    check_pair("{", "}", "braces")
    check_pair("[", "]", "brackets")

    # backticks: expect even count
    backticks = s.count("`")
    if backticks % 2 != 0:
        issues.append("unbalanced_backticks")

    return issues


def extract_entities(text: str):
    if text is None:
        text = ""
    urls = set(re.findall(URL_PATTERN, str(text)))
    emails = set(re.findall(EMAIL_PATTERN, str(text)))
    return urls, emails


def untranslated_terms(source: str, target: str) -> List[str]:
    """
    Very simple heuristic: if a key English term appears in the source AND
    appears literally in the target, we flag it as potentially untranslated.
    """
    if source is None or target is None:
        return []
    src = str(source)
    tgt = str(target)
    issues = []
    for term in KEY_TERMS:
        if term in src and term in tgt:
            issues.append(term)
    return issues


def apply_advanced_qa(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        src = row.get("source_text", "") or ""
        tgt = row.get("mt_output", "") or ""
        target_lang = row.get("target_lang", "") or ""

        src = str(src)
        tgt = str(tgt)
        target_lang = str(target_lang)

        nums_src = set(extract_numbers(src))
        nums_tgt = set(extract_numbers(tgt))
        ratio = length_ratio(src, tgt)
        nums_missing = nums_src - nums_tgt

        src_ph = extract_placeholders(src)
        tgt_ph = extract_placeholders(tgt)
        ph_missing = src_ph - tgt_ph

        src_p = sentence_final_punct(src)
        tgt_p = sentence_final_punct(tgt)

        struct_issues = balanced_brackets(tgt)

        src_urls, src_emails = extract_entities(src)
        tgt_urls, tgt_emails = extract_entities(tgt)

        missing_urls = src_urls - tgt_urls
        missing_emails = src_emails - tgt_emails

        untranslated = untranslated_terms(src, tgt)

        issues = []

        # Length ratio
        if ratio < 0.5 or ratio > 2.0:
            issues.append("length_ratio_out_of_range")

        # Missing numbers
        if nums_missing:
            issues.append(f"missing_numbers:{','.join(sorted(nums_missing))}")

        # Placeholders
        if ph_missing:
            issues.append(f"missing_placeholders:{','.join(sorted(ph_missing))}")

        # Punctuation
        if src_p == "?" and tgt_p != "?":
            issues.append("missing_question_mark")
        if src_p == "!" and tgt_p != "!":
            issues.append("missing_exclamation_mark")

        # CJK script for zh
        if target_lang == "zh" and not contains_cjk(tgt):
            issues.append("cjk_missing")

        # Structural issues
        issues.extend(struct_issues)

        # Entities
        if missing_urls:
            issues.append(f"missing_urls:{','.join(sorted(missing_urls))}")
        if missing_emails:
            issues.append(f"missing_emails:{','.join(sorted(missing_emails))}")

        # Untranslated terms
        if untranslated:
            issues.append(f"untranslated_terms:{'|'.join(untranslated)}")

        records.append(
            {
                "length_ratio": ratio,
                "numbers_src": ",".join(sorted(nums_src)) if nums_src else "",
                "numbers_tgt": ",".join(sorted(nums_tgt)) if nums_tgt else "",
                "qa_flags": ";".join(issues) if issues else "",
                "src_placeholders": ",".join(sorted(src_ph)) if src_ph else "",
                "tgt_placeholders": ",".join(sorted(tgt_ph)) if tgt_ph else "",
                "src_end_punct": src_p,
                "tgt_end_punct": tgt_p,
                "src_urls": ",".join(sorted(src_urls)) if src_urls else "",
                "tgt_urls": ",".join(sorted(tgt_urls)) if tgt_urls else "",
                "src_emails": ",".join(sorted(src_emails)) if src_emails else "",
                "tgt_emails": ",".join(sorted(tgt_emails)) if tgt_emails else "",
            }
        )

    qa_df = pd.DataFrame.from_records(records, index=df.index)
    return pd.concat([df, qa_df], axis=1)


# ----------------------
# Metrics
# ----------------------

def compute_metrics(df: pd.DataFrame):
    print("\n=== METRICS ===")

    # Average fluency score per language
    fluency_by_lang = df.groupby("target_lang")["fluency"].mean()
    print("\nAverage fluency score per language:")
    print(fluency_by_lang)

    # Frequency of placeholder errors
    placeholder_error_mask = df["qa_flags"].fillna("").str.contains("missing_placeholders")
    placeholder_error_count = placeholder_error_mask.sum()
    print(f"\nFrequency of placeholder errors: {placeholder_error_count} segments")

    # Percentage of segments with any QA flags
    any_flags_mask = df["qa_flags"].fillna("").astype(bool)
    pct_with_flags = (any_flags_mask.sum() / len(df)) * 100 if len(df) > 0 else 0.0
    print(f"\nPercentage of segments with any QA flags: {pct_with_flags:.2f}%")

    # Standard deviation of length_ratio per language
    std_length_by_lang = df.groupby("target_lang")["length_ratio"].std()
    print("\nStandard deviation of length_ratio per language:")
    print(std_length_by_lang)


# ----------------------
# Export CSV
# ----------------------

def export_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ----------------------
# Main
# ----------------------

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    print("Building dataset (10 examples × ES/DE/ZH)...")
    df = build_dataset()
    print(df.head(), "\n")

    print("Translating and back-translating...")
    df = add_mt_and_backtranslation(df)
    df = add_backtranslation_similarity(df)
    print(df[["id", "target_lang", "source_text", "mt_output", "back_translation_en", "backtranslation_similarity"]].head(), "\n")

    print("Applying advanced QA checks...")
    df = apply_advanced_qa(df)
    print(df[["id", "target_lang", "length_ratio", "qa_flags"]].head(), "\n")

    print("Running LLM-as-a-judge (MQM-style)...")
    df = add_llm_mqm_evaluations(df)
    print(df[["id", "target_lang", "adequacy", "fluency", "style_consistency", "overall_score"]].head(), "\n")

    print(f"Exporting full results to {OUTPUT_CSV}...")
    export_path = export_csv(df, OUTPUT_CSV)
    print(f"Saved: {export_path.resolve()}")

    compute_metrics(df)


if __name__ == "__main__":
    main()
