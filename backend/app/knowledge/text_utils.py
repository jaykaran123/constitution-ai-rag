"""Text normalisation shared by quote verification, BM25 and citation validation."""
from __future__ import annotations

import re
import unicodedata

from app.knowledge.constitution_parser import clean_text

_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'"}
_DASHES = {"—": "-", "–": "-", "‒": "-", "−": "-"}


def normalise_for_match(text: str) -> str:
    """Lower-case, unify quotes/dashes, drop amendment markers and brackets, collapse spaces."""
    text = (text or "").replace("<<<", " ").replace(">>>", " ")
    # Legal quotation style alters capitalisation in brackets: "[t]he State" quotes "The State".
    text = re.sub(r"\[([A-Za-z])\]", r"\1", text)
    text = unicodedata.normalize("NFKC", clean_text(text))
    for src, dst in {**_QUOTES, **_DASHES}.items():
        text = text.replace(src, dst)
    text = re.sub(r"-\s*\n\s*", "-", text)
    # Amendment brackets and omission stars are editorial: drop them without inserting spaces, so
    # "[the sovereignty and integrity of India], the security" matches "… of India, the security".
    text = re.sub(r"[\[\]*]", "", text)
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return text.strip().lower()


def contains_quote(haystack: str, quote: str) -> bool:
    return normalise_for_match(quote) in normalise_for_match(haystack)


STOP_WORDS = frozenset("""
a about after again all also am an and any are as at be because been before being between both but by can could
did do does for from had has have he her here him his how i if in into is it its me my of on or our please should
so some such than that the their them there these they this those through to under until up was we were what when
where which while who whom why will with would you your tell explain describe kindly shall may might
""".split())

_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "s")


def stem(word: str) -> str:
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            base = word[: -len(suffix)]
            if suffix in {"ies", "ied"}:
                base += "y"
            return base
    return word


def tokenize(text: str) -> list[str]:
    text = normalise_for_match(text)
    tokens = []
    for word in re.findall(r"[a-z]+|\d+[a-z]{0,3}", text):
        if word in STOP_WORDS or (len(word) < 3 and not word[0].isdigit()):
            continue
        tokens.append(stem(word))
    return tokens
