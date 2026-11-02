"""The small feedforward network and shared model inference helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


class MeanEmbeddingMLP(nn.Module):
    """Feedforward classifier over mean-pooled skip-gram word embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def mean_vectors(texts: Sequence[str], keyed_vectors, dim: int) -> np.ndarray:
    """Average available word vectors; use a zero vector when none are known."""
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        vectors = [keyed_vectors[w] for w in TOKEN_RE.findall(text) if w in keyed_vectors]
        if vectors:
            result[i] = np.mean(vectors, axis=0)
    return result


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


# File name of the INT8 student's quantized state_dict inside its artifact folder.
INT8_WEIGHTS = "model_int8.pt"


def quantize_int8(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of every nn.Linear layer (weights stored as int8,
    activations quantized on the fly at inference time). CPU only."""
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=torch.qint8).eval()


def count_int8_linears(model: nn.Module) -> int:
    """Number of dynamically quantized Linear layers in a model."""
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear
    return sum(isinstance(module, DynamicQuantizedLinear) for module in model.modules())


def load_int8_student(folder) -> nn.Module:
    """Rebuild the INT8 student saved by src/train.py.

    The quantized model is saved as a state_dict (not a pickled module). To load it we
    create an FP32 student with the saved config, apply the same dynamic quantization
    so the module structure matches, and then load the int8 weights into it.
    """
    from pathlib import Path
    from transformers import BertConfig, BertForSequenceClassification

    folder = Path(folder)
    config = BertConfig.from_pretrained(folder)
    model = quantize_int8(BertForSequenceClassification(config))
    state = torch.load(folder / INT8_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()

