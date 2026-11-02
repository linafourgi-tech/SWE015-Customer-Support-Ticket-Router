"""Train and benchmark the four-model SWE015 intent-routing ladder.

Run from the project root with: python -m src.train
All reported metrics come from this script's local test split evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import datasets
import gensim
import joblib
import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import Pipeline
import sklearn
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
)
from .data import load_clinc, normalize
from .models import (
    INT8_WEIGHTS,
    MeanEmbeddingMLP,
    count_int8_linears,
    load_int8_student,
    mean_vectors,
    probabilities,
    quantize_int8,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BASE_CHECKPOINT = "bert-base-uncased"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def folder_size_mb(path: Path) -> float:
    files = [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / (1024 * 1024)


def trim_batch(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop padding columns that are empty for the whole batch.

    The datasets are tokenized once with padding to the longest training sentence. BERT
    pads on the right, so cutting every batch to its own longest sentence gives the same
    outputs while making CPU training much faster for short messages.
    """
    length = int(mask.sum(dim=1).max())
    return ids[:, :length], mask[:, :length]


def hardware_info() -> dict:
    """CPU, RAM, OS and library versions of the machine that produced the measurements."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_quantized_engine": torch.backends.quantized.engine,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu",
    }
    if platform.system() == "Windows":
        try:
            import ctypes
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu_name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info["ram_gb"] = round(status.ullTotalPhys / 1024**3, 1)
        except OSError:
            pass
    info.update({
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "scikit_learn_version": sklearn.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "gensim_version": gensim.__version__,
    })
    return info


def benchmark(predict_one, texts: list[str], labels: list[str],
              oos_label: str) -> tuple[dict, list[str], list[float]]:
    """Measure accuracy/OOS recall and end-to-end latency on identical inputs.

    Returns the metrics row plus the per-example predictions and confidences.
    """
    pred_labels: list[str] = []
    confidences: list[float] = []
    for text in texts:
        label, confidence = predict_one(text)
        pred_labels.append(label)
        confidences.append(float(confidence))

    timings: list[float] = []
    sample = texts[: min(100, len(texts))]
    for text in sample[:10]:
        predict_one(text)
    # Cycle through held-out samples until at least 100 timed calls are recorded.
    for i in range(max(100, len(sample))):
        text = sample[i % len(sample)]
        started = time.perf_counter()
        predict_one(text)
        timings.append((time.perf_counter() - started) * 1000)

    row = {
        "accuracy": float(accuracy_score(labels, pred_labels)),
        "out_of_scope_recall": float(
            recall_score(labels, pred_labels, labels=[oos_label], average="macro", zero_division=0)
        ),
        "latency_ms_mean": float(statistics.mean(timings)),
        "latency_ms_std": float(statistics.pstdev(timings)),
        "latency_runs": len(timings),
        "test_examples": len(labels),
    }
    return row, pred_labels, confidences


def train_mlp(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
              num_classes: int, epochs: int, seed: int) -> MeanEmbeddingMLP:
    torch.manual_seed(seed)
    model = MeanEmbeddingMLP(train_x.shape[1], 128, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    best_state = None
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(torch.from_numpy(val_x)), torch.from_numpy(val_y)).item()
        print(f"MLP epoch {epoch + 1}/{epochs}: validation_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model.eval()


def create_student_model(teacher: BertForSequenceClassification, num_layers: int = 4):
    """Create a smaller BERT student (default 4 layers) initialized from the teacher."""
    import copy

    teacher_layers = teacher.config.num_hidden_layers
    if not 0 < num_layers < teacher_layers:
        raise ValueError(f"Student must have between 1 and {teacher_layers - 1} layers, got {num_layers}.")

    # Same hidden size, vocabulary, label count and label mappings as the teacher;
    # only the depth of the Transformer encoder changes.
    student_config = copy.deepcopy(teacher.config)
    student_config.num_hidden_layers = num_layers

    student = BertForSequenceClassification(student_config)

    # Copy the teacher's learned token embeddings.
    student.bert.embeddings.load_state_dict(
        teacher.bert.embeddings.state_dict()
    )

    # Copy the first `num_layers` Transformer layers.
    for i in range(num_layers):
        student.bert.encoder.layer[i].load_state_dict(
            teacher.bert.encoder.layer[i].state_dict()
        )

    # Copy the classification head.
    student.classifier.load_state_dict(
        teacher.classifier.state_dict()
    )

    if teacher.bert.pooler is not None:
        student.bert.pooler.load_state_dict(
            teacher.bert.pooler.state_dict()
        )

    return student.cpu()


def train_student(
    teacher,
    student,
    train_ids,
    train_mask,
    train_labels,
    val_ids,
    val_mask,
    val_labels,
    epochs=3,
    temperature=2.0,
    alpha=0.5,
):
    """Train the smaller student using teacher predictions and true labels."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=3e-5, weight_decay=0.01
    )
    hard_loss_fn = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        train_ids, train_mask, torch.from_numpy(train_labels)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        student.train()

        for ids, mask, targets in train_loader:
            ids, mask = trim_batch(ids, mask)
            optimizer.zero_grad(set_to_none=True)

            # The fine-tuned teacher is frozen: it only provides soft targets.
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=ids, attention_mask=mask
                ).logits

            student_logits = student(
                input_ids=ids, attention_mask=mask
            ).logits

            soft_loss = nn.functional.kl_div(
                nn.functional.log_softmax(
                    student_logits / temperature, dim=-1
                ),
                nn.functional.softmax(
                    teacher_logits / temperature, dim=-1
                ),
                reduction="batchmean",
            ) * (temperature ** 2)

            hard_loss = hard_loss_fn(student_logits, targets)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        student.eval()
        val_losses = []

        with torch.no_grad():
            for start in range(0, len(val_labels), 64):
                end = start + 64
                ids, mask = trim_batch(val_ids[start:end], val_mask[start:end])
                targets = torch.from_numpy(val_labels[start:end])

                teacher_logits = teacher(
                    input_ids=ids, attention_mask=mask
                ).logits

                student_logits = student(
                    input_ids=ids, attention_mask=mask
                ).logits

                soft_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(
                        student_logits / temperature, dim=-1
                    ),
                    nn.functional.softmax(
                        teacher_logits / temperature, dim=-1
                    ),
                    reduction="batchmean",
                ) * (temperature ** 2)

                hard_loss = hard_loss_fn(student_logits, targets)
                val_loss = alpha * soft_loss + (1 - alpha) * hard_loss
                val_losses.append(val_loss.item())

        mean_val_loss = float(np.mean(val_losses))
        print(
            f"Student epoch {epoch + 1}/{epochs}: "
            f"validation_loss={mean_val_loss:.4f}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in student.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Student training did not produce a valid checkpoint.")

    student.load_state_dict(best_state)
    return student.eval()


def train_models(args, texts, labels_by_split, labels: list[str]) -> dict[str, float]:
    """Train all models and save them under artifacts/. Returns minutes per stage.

    Everything created here (optimizers, best-epoch weight copies, in-memory models) is
    released when this function returns, so evaluation starts with free memory.
    """
    stage_minutes: dict[str, float] = {}
    train_texts, val_texts = texts["train"], texts["validation"]
    train_labels, val_labels, _ = labels_by_split
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y_train = np.asarray([label_to_id[x] for x in train_labels], dtype=np.int64)
    y_val = np.asarray([label_to_id[x] for x in val_labels], dtype=np.int64)

    stage_started = time.perf_counter()
    print("Training TF-IDF + logistic regression baseline...")
    classical = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("logreg", LogisticRegression(max_iter=400, C=4.0, solver="lbfgs")),
    ])
    classical.fit(train_texts, train_labels)
    joblib.dump(classical, ARTIFACTS / "classical" / "model.joblib")
    stage_minutes["tfidf_logistic_regression"] = (time.perf_counter() - stage_started) / 60

    stage_started = time.perf_counter()
    print("Training skip-gram embeddings on training messages...")
    tokenized = [text.split() for text in train_texts]
    w2v = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1,
                   workers=1, sg=1, negative=5, epochs=8, seed=args.seed)
    w2v.wv.save(str(ARTIFACTS / "mlp" / "vectors.kv"))
    x_train = mean_vectors(train_texts, w2v.wv, 100)
    x_val = mean_vectors(val_texts, w2v.wv, 100)
    mlp = train_mlp(x_train, y_train, x_val, y_val, len(labels), args.epochs, args.seed)
    torch.save({"state_dict": mlp.state_dict(), "embedding_dim": 100,
                "hidden_dim": 128, "num_classes": len(labels)}, ARTIFACTS / "mlp" / "model.pt")
    stage_minutes["skipgram_mean_embedding_mlp"] = (time.perf_counter() - stage_started) / 60

    stage_started = time.perf_counter()
    print(f"Fine-tuning {args.bert_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model, num_labels=len(labels), label2id=label_to_id, id2label=id_to_label
    ).cpu()
    encoded_train = tokenizer(train_texts, truncation=True, padding=True,
                              max_length=args.max_length, return_tensors="pt")
    encoded_val = tokenizer(val_texts, truncation=True, padding=True,
                            max_length=args.max_length, return_tensors="pt")
    train_ds = TensorDataset(encoded_train["input_ids"], encoded_train["attention_mask"],
                             torch.from_numpy(y_train))
    val_ids, val_mask = encoded_val["input_ids"], encoded_val["attention_mask"]
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
    best_state = None
    best_val = float("inf")
    for epoch in range(args.bert_epochs):
        bert.train()
        for ids, mask, target in loader:
            ids, mask = trim_batch(ids, mask)
            optimizer.zero_grad(set_to_none=True)
            loss = bert(input_ids=ids, attention_mask=mask, labels=target).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            optimizer.step()
        bert.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(y_val), 64):
                end = start + 64
                ids, mask = trim_batch(val_ids[start:end], val_mask[start:end])
                loss = bert(input_ids=ids, attention_mask=mask,
                            labels=torch.from_numpy(y_val[start:end])).loss
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        print(f"BERT epoch {epoch + 1}/{args.bert_epochs}: validation_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in bert.state_dict().items()}
    if best_state:
        bert.load_state_dict(best_state)
    bert.save_pretrained(ARTIFACTS / "bert")
    tokenizer.save_pretrained(ARTIFACTS / "bert")
    # The optimizer state (~2x model size) and the weight copy are no longer needed.
    del optimizer, best_state
    gc.collect()
    stage_minutes["bert_teacher_finetuning"] = (time.perf_counter() - stage_started) / 60

    # Create and train the smaller student using knowledge distillation.
    # The teacher above is already fine-tuned; it is frozen inside train_student().
    stage_started = time.perf_counter()
    print(f"Creating the {args.student_layers}-layer BERT student...")
    student = create_student_model(bert, args.student_layers)

    print("Training student using knowledge distillation...")
    student = train_student(
        teacher=bert,
        student=student,
        train_ids=encoded_train["input_ids"],
        train_mask=encoded_train["attention_mask"],
        train_labels=y_train,
        val_ids=val_ids,
        val_mask=val_mask,
        val_labels=y_val,
        epochs=args.student_epochs,
    )

    # Save the best (lowest validation loss) FP32 student before quantization.
    student.save_pretrained(ARTIFACTS / "student")
    tokenizer.save_pretrained(ARTIFACTS / "student")
    stage_minutes["student_distillation"] = (time.perf_counter() - stage_started) / 60

    # Quantize the distilled student (not the teacher) to INT8 for CPU inference.
    # Only the quantized state_dict is saved; src.models.load_int8_student() rebuilds
    # the module from config.json, which is more robust than pickling the whole module.
    print("Quantizing the distilled student to INT8...")
    int8_dir = ARTIFACTS / "student_int8"
    torch.save(quantize_int8(student).state_dict(), int8_dir / INT8_WEIGHTS)
    student.config.save_pretrained(int8_dir)
    tokenizer.save_pretrained(int8_dir)

    # Record the settings used, so a later --evaluate-only run reports them correctly.
    (ARTIFACTS / "training_config.json").write_text(json.dumps({
        "seed": args.seed, "mlp_epochs": args.epochs, "bert_checkpoint": args.bert_model,
        "bert_epochs": args.bert_epochs, "max_length": args.max_length,
        "student_layers": args.student_layers, "student_epochs": args.student_epochs,
        "stage_minutes": {k: round(v, 2) for k, v in stage_minutes.items()},
    }, indent=2), encoding="utf-8")
    return stage_minutes


def evaluate_saved_models(args, test_texts: list[str], test_labels: list[str],
                          labels: list[str], oos_label: str, split_sizes: dict) -> dict:
    """Load each saved model from artifacts/ (as the app does), benchmark it on the
    test split, then free it before loading the next one to keep memory low."""
    id_to_label = dict(enumerate(labels))
    int8_dir = ARTIFACTS / "student_int8"
    tokenizer = AutoTokenizer.from_pretrained(int8_dir)

    # Every predictor receives the raw test string and applies the same normalize()
    # used by the app, so the timed latency covers preprocessing + tokenization +
    # inference. (The test texts are already normalized; normalize() is idempotent.)
    def bert_predictor(model):
        def predict(text: str) -> tuple[str, float]:
            inp = tokenizer(normalize(text), truncation=True, max_length=args.max_length, return_tensors="pt")
            with torch.no_grad():
                p = probabilities(model(**inp).logits)[0].numpy()
            i = int(np.argmax(p))
            return id_to_label[i], float(p[i])
        return predict

    def load_classical():
        classical = joblib.load(ARTIFACTS / "classical" / "model.joblib")

        def predict(text: str) -> tuple[str, float]:
            probs = classical.predict_proba([normalize(text)])[0]
            i = int(np.argmax(probs))
            return str(classical.classes_[i]), float(probs[i])
        return predict, {}

    def load_mlp():
        vectors = KeyedVectors.load(str(ARTIFACTS / "mlp" / "vectors.kv"))
        checkpoint = torch.load(ARTIFACTS / "mlp" / "model.pt", map_location="cpu", weights_only=True)
        mlp = MeanEmbeddingMLP(checkpoint["embedding_dim"], checkpoint["hidden_dim"], checkpoint["num_classes"])
        mlp.load_state_dict(checkpoint["state_dict"])
        mlp.eval()

        def predict(text: str) -> tuple[str, float]:
            x = mean_vectors([normalize(text)], vectors, vectors.vector_size)
            with torch.no_grad():
                p = probabilities(mlp(torch.from_numpy(x)))[0].numpy()
            i = int(np.argmax(p))
            return id_to_label[i], float(p[i])
        return predict, {}

    def load_fp32(folder: str):
        def loader():
            model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS / folder).cpu().eval()
            info = {"layers": model.config.num_hidden_layers,
                    "parameters": sum(p.numel() for p in model.parameters())}
            return bert_predictor(model), info
        return loader

    def load_int8():
        model = load_int8_student(int8_dir)
        int8_linears = count_int8_linears(model)
        if int8_linears == 0:
            raise RuntimeError("Quantization failed: no INT8 Linear layers found in the reloaded student.")
        print(f"Reloaded INT8 student contains {int8_linears} quantized Linear layers.")
        return bert_predictor(model), {"layers": model.config.num_hidden_layers,
                                       "int8_linear_layers": int8_linears}

    # (group, name, loader, files counted in size_mb, description of size_mb)
    # "models" is the four-model ladder required by the assignment. The supplementary
    # FP32 student is diagnostic only: it separates the effect of distillation
    # (teacher -> FP32 student) from the effect of quantization (FP32 -> INT8 student).
    evaluations = [
        ("models", "tfidf_logistic_regression", load_classical, ARTIFACTS / "classical" / "model.joblib",
         "classical/model.joblib (TF-IDF vocabulary + logistic regression weights)"),
        ("models", "skipgram_mean_embedding_mlp", load_mlp, ARTIFACTS / "mlp",
         "mlp/ folder (Word2Vec vectors + MLP weights)"),
        ("models", "bert_finetuned_fp32", load_fp32("bert"), ARTIFACTS / "bert",
         "bert/ folder (FP32 safetensors weights + config + tokenizer files)"),
        ("models", "bert_student_int8", load_int8, int8_dir,
         "student_int8/ folder (INT8 state_dict + config + tokenizer files)"),
        ("supplementary_models", "bert_student_fp32", load_fp32("student"), ARTIFACTS / "student",
         "student/ folder (FP32 safetensors weights + config + tokenizer files)"),
    ]

    config_path = ARTIFACTS / "training_config.json"
    if config_path.exists():
        train_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg_source = "artifacts/training_config.json"
    else:
        train_cfg = {"seed": args.seed, "mlp_epochs": args.epochs, "bert_checkpoint": args.bert_model,
                     "bert_epochs": args.bert_epochs, "max_length": args.max_length,
                     "student_layers": args.student_layers, "student_epochs": args.student_epochs}
        cfg_source = "command-line arguments of the evaluation run (training_config.json not found)"
    teacher_cfg = BertConfig.from_pretrained(ARTIFACTS / "bert")
    student_cfg = BertConfig.from_pretrained(int8_dir)

    result = {
        "dataset": "CLINC150 plus",
        **split_sizes,
        "label_count": len(labels),
        "out_of_scope_label": oos_label,
        "training_config": train_cfg,
        "training_config_source": cfg_source,
        "distillation": {
            "teacher_layers": teacher_cfg.num_hidden_layers,
            "student_layers": student_cfg.num_hidden_layers,
            "student_init": "teacher embeddings + first N encoder layers + pooler + classifier",
            "temperature": 2.0,
            "alpha_soft": 0.5,
            "loss": "alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * cross_entropy(labels)",
        },
        "quantization": "torch.ao.quantization.quantize_dynamic, all nn.Linear layers, qint8, applied to the distilled student",
        "latency_note": ("CPU, batch size 1, one raw string per call including normalize() + tokenization + "
                         "forward pass + softmax; 10 warm-ups then >=100 timed calls on the first 100 test messages"),
        "size_note": "Size on disk of the files needed to run each model; see size_includes per model.",
        "hardware": hardware_info(),
        "models": {},
        "supplementary_models": {},
    }
    for group, name, loader, size_path, size_text in evaluations:
        print(f"Evaluating {name}...")
        predictor, info = loader()
        row, predicted, confidences = benchmark(predictor, test_texts, test_labels, oos_label)
        row["size_mb"] = folder_size_mb(size_path) if size_path.is_dir() else size_path.stat().st_size / (1024 * 1024)
        row["size_includes"] = size_text
        row.update(info)
        if name == "bert_student_int8":
            # Real misclassified test messages from the application model (for error analysis).
            row["example_errors"] = [
                {"text": text, "true": true, "predicted": pred, "confidence": round(conf, 3)}
                for text, true, pred, conf in zip(test_texts, test_labels, predicted, confidences)
                if pred != true
            ][:25]
        result[group][name] = row
        print(json.dumps({name: {k: v for k, v in row.items() if k != "example_errors"}}, indent=2))
        del predictor
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    # 5 epochs underfit (validation accuracy ~0.50 vs ~0.71 at 30); the best epoch is
    # still selected by validation loss inside train_mlp().
    parser.add_argument("--epochs", type=int, default=30, help="MLP epochs")
    parser.add_argument("--bert-epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--bert-model", default=BASE_CHECKPOINT)
    parser.add_argument("--student-layers", type=int, default=4)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training; benchmark the models already saved in artifacts/")
    args = parser.parse_args()
    set_seed(args.seed)
    run_started = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in ("classical", "mlp", "bert", "student", "student_int8"):
        (ARTIFACTS / name).mkdir(parents=True, exist_ok=True)

    texts, labels_by_split = load_clinc()
    train_labels, val_labels, test_labels = labels_by_split
    labels = sorted(set(train_labels))
    unknown = set(val_labels + test_labels) - set(labels)
    if unknown:
        raise RuntimeError(f"Validation/test labels missing from training labels: {sorted(unknown)}")
    oos_label = next((x for x in labels if x.lower() in {"oos", "out_of_scope"}), None)
    if oos_label is None:
        raise RuntimeError(f"Could not find the CLINC out-of-scope label in: {labels[-5:]}")

    labels_path = ARTIFACTS / "labels.json"
    if args.evaluate_only:
        if not labels_path.exists() or json.loads(labels_path.read_text(encoding="utf-8")) != labels:
            raise RuntimeError("artifacts/labels.json is missing or does not match the dataset. Train first.")
        stage_minutes: dict[str, float] = {}
    else:
        labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        stage_minutes = train_models(args, texts, labels_by_split, labels)
        gc.collect()

    stage_started = time.perf_counter()
    split_sizes = {"train_examples": len(train_labels), "validation_examples": len(val_labels),
                   "test_examples": len(test_labels)}
    result = evaluate_saved_models(args, texts["test"], test_labels, labels, oos_label, split_sizes)
    stage_minutes["evaluation"] = (time.perf_counter() - stage_started) / 60
    stage_minutes["total_this_run"] = (time.perf_counter() - run_started) / 60
    result["run_mode"] = "evaluate-only (models loaded from artifacts/)" if args.evaluate_only else "train + evaluate"
    result["wall_clock_minutes_this_run"] = {k: round(v, 2) for k, v in stage_minutes.items()}
    (ARTIFACTS / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to {ARTIFACTS / 'results.json'}")


if __name__ == "__main__":
    main()
