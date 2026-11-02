"""Streamlit interface for the support ticket router (distilled INT8 BERT student)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
from gensim.models import KeyedVectors
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import extract_model_code, normalize
from src.models import MeanEmbeddingMLP, count_int8_linears, load_int8_student, mean_vectors, probabilities


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
STUDENT_INT8_DIR = ARTIFACTS / "student_int8"
PRODUCTS = ["dishwasher", "washing machine", "refrigerator", "oven", "microwave", "dryer", "air conditioner"]
GREETINGS = re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b", re.I)
THANKS = re.compile(r"\b(thanks|thank you|thx)\b", re.I)
# Words that may accompany a greeting/thanks without making it a real support request.
FILLER = {"hi", "hello", "hey", "good", "morning", "afternoon", "evening", "thanks", "thank",
          "you", "thx", "so", "much", "a", "lot", "there", "again", "very", "ok", "okay"}
# ELIZA-style patterns: match a feeling or statement and reflect it back to the user.
ELIZA_RULES = [
    (re.compile(r"\bi(?: am|'m) (?:so |very |really )?(frustrated|angry|annoyed|upset|confused|worried|sad|tired)\b", re.I),
     "I'm sorry you are feeling {0}. Let's see what we can do about it."),
    (re.compile(r"\bi feel (.+?)(?:[.!?,]|$)", re.I), "Why do you feel {0}?"),
]
REFLECTIONS = {"i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you're", "myself": "yourself"}
DEVICE = "cpu"
# Display names for the four-model ladder; the last one is the application model.
APP_MODEL = "Distilled BERT student (4-layer, INT8)"


@st.cache_resource
def load_models():
    labels = json.loads((ARTIFACTS / "labels.json").read_text(encoding="utf-8"))
    classical = joblib.load(ARTIFACTS / "classical" / "model.joblib")
    vectors = KeyedVectors.load(str(ARTIFACTS / "mlp" / "vectors.kv"), mmap="r")
    checkpoint = torch.load(ARTIFACTS / "mlp" / "model.pt", map_location=DEVICE, weights_only=True)
    mlp = MeanEmbeddingMLP(checkpoint["embedding_dim"], checkpoint["hidden_dim"], checkpoint["num_classes"])
    mlp.load_state_dict(checkpoint["state_dict"])
    mlp.eval()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_INT8_DIR)
    teacher = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS / "bert").cpu().eval()
    student = load_int8_student(STUDENT_INT8_DIR)
    # Guard against loading the wrong artifact (e.g. a directly quantized teacher).
    student_layers = student.config.num_hidden_layers
    if student_layers >= teacher.config.num_hidden_layers or count_int8_linears(student) == 0:
        raise RuntimeError("artifacts/student_int8 does not contain a distilled INT8 student. Re-run training.")
    if student.config.num_labels != len(labels) or teacher.config.num_labels != len(labels):
        raise RuntimeError("Label count in labels.json does not match the BERT models. Re-run training.")
    return labels, classical, vectors, mlp, tokenizer, teacher, student


def route_team(intent: str) -> str:
    """Map predicted CLINC category-like prefixes to an interpretable queue."""
    if intent == "oos":
        return "Human agent"
    families = {
        "bank": "Billing and payments", "credit": "Billing and payments", "cash": "Billing and payments",
        "card": "Billing and payments", "pay": "Billing and payments", "bill": "Billing and payments",
        "travel": "Orders and travel", "car_": "Orders and travel", "flight": "Orders and travel",
        "hotel": "Orders and travel", "book_": "Orders and travel", "restaurant": "Orders and travel",
        "calendar": "Account and scheduling", "alarm": "Account and scheduling", "reminder": "Account and scheduling",
        "email": "Account and digital services", "phone": "Account and digital services", "music": "Account and digital services",
        "weather": "General information", "timer": "General information", "time": "General information",
    }
    for prefix, team in families.items():
        if intent.startswith(prefix):
            return team
    return "General support"


def extract_product(text: str) -> tuple[str | None, str | None]:
    words = re.findall(r"[a-z]+", text.lower())
    corrected = None
    product_tokens = {token: product for product in PRODUCTS for token in product.split()}
    for word in words:
        if len(word) < 5:
            continue
        distances = [(edit_distance(word, token), product) for token, product in product_tokens.items()]
        distance, candidate = min(distances)
        if distance == 1 and word not in product_tokens:
            corrected = candidate
            break
    product = corrected or next((p for p in PRODUCTS if p in text.lower()), None)
    # Same model-code rule as the training normalizer (src/data.py).
    return product, extract_model_code(text)


def correct_product_text(text: str) -> tuple[str, str | None, str | None]:
    """Correct a one-edit product-token typo and return product/code metadata."""
    product, code = extract_product(text)
    if not product:
        return text, None, code
    product_tokens = {token for name in PRODUCTS for token in name.split()}
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) < 5 or word in product_tokens:
            continue
        nearest = min(product_tokens, key=lambda token: edit_distance(word, token))
        if edit_distance(word, nearest) == 1:
            fixed = re.sub(rf"\b{re.escape(word)}\b", nearest, text, count=1, flags=re.I)
            return fixed, product, code
    return text, product, code


def edit_distance(source: str, target: str) -> int:
    """Levenshtein distance using insertions, deletions and substitutions at unit cost."""
    previous = list(range(len(target) + 1))
    for i, left in enumerate(source, start=1):
        current = [i]
        for j, right in enumerate(target, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def reflect(fragment: str) -> str:
    """Swap first-person words for second-person ones, as in ELIZA ("my oven" -> "your oven")."""
    return " ".join(REFLECTIONS.get(word.lower(), word) for word in fragment.split())


def small_talk_reply(text: str) -> tuple[str | None, bool]:
    """Rule-based ELIZA-style reply.

    Returns (reply, only_small_talk). only_small_talk is True when the message contains
    nothing except greeting/thanks words, so it does not need to be routed to a team.
    A greeting followed by a real request ("hi, my card was declined") is still routed.
    """
    words = re.findall(r"[a-z']+", text.lower())
    only_small_talk = bool(words) and all(word in FILLER for word in words)
    if GREETINGS.search(text.strip()):
        return "Hello! How can I help you today?", only_small_talk
    if THANKS.search(text):
        return "You're welcome. Is there anything else I can help with?", only_small_talk
    for pattern, template in ELIZA_RULES:
        match = pattern.search(text)
        if match:
            return template.format(reflect(match.group(1))), False
    return None, False


def predict_all(text: str, labels, classical, vectors, mlp, tokenizer, teacher, student):
    clean = normalize(text)
    results = {}
    p = classical.predict_proba([clean])[0]
    i = int(np.argmax(p))
    results["TF-IDF + logistic regression"] = (str(classical.classes_[i]), float(p[i]))
    features = mean_vectors([clean], vectors, vectors.vector_size)
    with torch.no_grad():
        p = probabilities(mlp(torch.from_numpy(features)))[0].numpy()
    i = int(np.argmax(p))
    results["Skip-gram + feedforward network"] = (labels[i], float(p[i]))
    inputs = tokenizer(clean, truncation=True, max_length=96, return_tensors="pt")
    with torch.no_grad():
        p = probabilities(teacher(**inputs).logits)[0].numpy()
    i = int(np.argmax(p))
    results["Fine-tuned BERT teacher (12-layer, FP32)"] = (labels[i], float(p[i]))
    with torch.no_grad():
        p = probabilities(student(**inputs).logits)[0].numpy()
    i = int(np.argmax(p))
    results[APP_MODEL] = (labels[i], float(p[i]))
    return clean, results


st.set_page_config(page_title="Support ticket router", page_icon="🛠️", layout="wide")
st.title("Support Ticket Router")
st.caption("SWE015 project · CLINC150 intent benchmark · CPU inference")

if not (ARTIFACTS / "results.json").exists():
    st.warning("Models have not been trained yet. From this folder, run: `python -m src.train`")
    st.stop()

try:
    labels, classical, vectors, mlp, tokenizer, teacher, student = load_models()
except Exception as exc:
    st.error(f"Could not load local model artifacts from `artifacts/`: {exc}  \n"
             "Re-run `python -m src.train` to regenerate them.")
    st.stop()

threshold = st.sidebar.slider("Human-agent confidence threshold", 0.10, 0.95, 0.45, 0.05)
show_all = st.sidebar.checkbox("Compare model predictions side by side", True)
message = st.text_area("Customer message", placeholder="Example: I need help with my card payment", height=120)

if st.button("Route message", type="primary", disabled=not message.strip()):
    corrected_message, product, code = correct_product_text(message)
    reply, only_small_talk = small_talk_reply(message)
    try:
        clean, predictions = predict_all(corrected_message, labels, classical, vectors, mlp,
                                         tokenizer, teacher, student)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()
    # Routing decisions use only the distilled INT8 student; other models are for comparison.
    selected_intent, confidence = predictions[APP_MODEL]
    needs_agent = not only_small_talk and (selected_intent == "oos" or confidence < threshold)
    if only_small_talk:
        team = "Automated small talk"
    else:
        team = "Human agent" if needs_agent else route_team(selected_intent)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Routing result")
        st.metric("Team", team)
        st.write(f"**Predicted intent:** `{selected_intent}`")
        st.write(f"**Confidence:** {confidence:.1%}")
        if corrected_message != message:
            st.write(f"**Product-name correction:** {corrected_message}")
        st.write(f"**Normalized text:** {clean}")
        if product or code:
            st.write(f"**Detected product:** {product or '—'} · **Model code:** {code or '—'}")
        if needs_agent:
            st.info("Low confidence or out-of-scope: send this message to a human agent.")
        if reply:
            st.success(f"Rule-based small-talk reply: {reply}")
    with right:
        st.subheader("Application model")
        st.write(f"{student.config.num_hidden_layers}-layer BERT student distilled from the fine-tuned "
                 f"{teacher.config.num_hidden_layers}-layer bert-base-uncased teacher, with dynamic INT8 "
                 "quantization of its Linear layers, running on CPU.")
        st.caption("Intents come from the CLINC150 label set (general assistant intents, not "
                   "appliance faults). Product names and model codes are extracted by rules only.")

    if show_all:
        st.subheader("Model comparison")
        rows = [{"Model": name, "Intent": intent, "Confidence": f"{score:.1%}"}
                for name, (intent, score) in predictions.items()]
        st.dataframe(rows, hide_index=True, width="stretch")

st.divider()
st.caption("Research prototype. Benchmark results are produced by the training script on the local machine; see artifacts/results.json.")
