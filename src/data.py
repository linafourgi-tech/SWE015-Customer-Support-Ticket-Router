"""Load the public CLINC150 benchmark and apply lightweight normalization."""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset


# Product model codes such as "wm3500" or "WM-3500" (any letter case).
_MODEL_CODE = re.compile(r"\b([a-z]{1,5})-?(\d{2,6})\b", re.IGNORECASE)
# Space-separated codes ("WM 3500") are only accepted when the letters are typed in
# upper case. Allowing lower-case words here would glue ordinary words to numbers
# ("send 100" -> "send100", "for 20" -> "for20") in about 2.7% of CLINC150 texts.
_SPACED_MODEL_CODE = re.compile(r"\b([A-Z]{2,5}) (\d{2,6})\b")
_REPEATED = re.compile(r"(.)\1{2,}")
_NON_WORD = re.compile(r"[^\w\s'-]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def extract_model_code(text: str) -> str | None:
    """Return the first product model code in canonical form (e.g. "WM3500"), if any."""
    match = _SPACED_MODEL_CODE.search(text) or _MODEL_CODE.search(text)
    return f"{match.group(1)}{match.group(2)}".upper() if match else None


def normalize(text: str) -> str:
    """Normalize noisy input while preserving useful product-code tokens."""
    text = _SPACED_MODEL_CODE.sub(lambda m: f" {m.group(1)}{m.group(2)} ", text.strip())
    text = text.lower()
    text = _MODEL_CODE.sub(lambda m: f" {m.group(1)}{m.group(2)} ", text)
    text = _REPEATED.sub(r"\1\1", text)
    text = _NON_WORD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def load_clinc() -> tuple[dict[str, list[str]], list[list[str]]]:
    """Return train/validation/test texts and labels as strings."""
    ds = load_dataset("clinc_oos", "plus")
    feature = ds["train"].features["intent"]

    def label_name(value: Any) -> str:
        if hasattr(feature, "int2str"):
            return feature.int2str(int(value))
        return str(value)

    splits: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        rows = ds[split]
        splits[split] = [normalize(row["text"]) for row in rows]
        if split == "train":
            labels = [label_name(row["intent"]) for row in rows]
        else:
            split_labels = [label_name(row["intent"]) for row in rows]
            if split == "test":
                test_labels = split_labels
            else:
                validation_labels = split_labels

    # Keep labels in the same ordered split structure as texts.
    label_splits = [labels, validation_labels, test_labels]
    return splits, label_splits

