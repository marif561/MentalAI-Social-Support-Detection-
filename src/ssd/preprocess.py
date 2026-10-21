"""Text cleaning used for all subtasks (paper Section 3.2).

Lower-casing, removal of URLs, user mentions and unnecessary special characters, and
whitespace normalisation. Letters with accents (Spanish), digits, emojis, hashtags'
words and basic punctuation are kept, since transformers benefit from raw social-media
style text.
"""
from __future__ import annotations

import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
# characters treated as noise: anything that is not a word char, whitespace,
# basic punctuation, or an emoji / symbol in the supplementary planes
SPECIAL_RE = re.compile(r"[^\w\s.,!?¿¡'\"#:;()\-\U0001F000-\U0001FAFF☀-➿]")
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = SPECIAL_RE.sub(" ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1\1", text)
    return WS_RE.sub(" ", text).strip()
