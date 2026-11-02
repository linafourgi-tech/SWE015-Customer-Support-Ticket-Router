# Support Ticket Intent Router (SWE015, Use Case 2)

An educational, CPU-only intent-routing application built on the CLINC150 benchmark. Four models are trained and evaluated on the **same held-out test split**:

1. TF-IDF + logistic regression
2. Skip-gram Word2Vec (mean-pooled) + feedforward MLP trained with SGD/backpropagation
3. Fine-tuned `bert-base-uncased` teacher (12 layers, FP32)
4. **Distilled 4-layer BERT student, dynamically quantized to INT8** — this is the model the Streamlit app uses for routing

The app shows all four predictions side by side, but routing decisions use only the distilled INT8 student.

## Requirements

- Python 3.12 (verified with 3.12.10 on Windows 11). Python 3.10/3.11 should also work with the same requirements, but they were not tested.
- CPU only; no GPU or CUDA needed (the code always runs on `cpu`).
- About 3 GB of free RAM for BERT fine-tuning, and about 1.5 GB of disk for artifacts plus the Hugging Face cache.
- Internet access for setup (cloning, `pip install`, downloading the artifact archive). Training and `--evaluate-only` also need internet (they download CLINC150 and, for training, `bert-base-uncased` from Hugging Face). The app itself runs offline once the artifacts are in place.

Verified package versions: PyTorch 2.9.1+cpu, Transformers 4.57.6, Datasets 4.8.5, scikit-learn 1.9.1, gensim 4.4.0, NumPy 2.5.4, Streamlit 1.66.1.

PyTorch is capped below 2.10 because `torch.ao.quantization.quantize_dynamic` (used for the INT8 student) is deprecated and scheduled for removal in 2.10.

## Quick start from a fresh clone (Windows PowerShell)

The trained models are not stored in Git. They are published as a release asset, so the app can be run **without retraining**:

- Release: https://github.com/linafourgi-tech/SWE015-Customer-Support-Ticket-Router/releases/tag/v1.0.0
- Asset: `support-ticket-router-artifacts.zip` (723,993,557 bytes, about 690 MB)
- SHA-256: `f4779dc3eac1c8c8775c8534593957740a646e9d3bd0ebc240ba14a4b246a210`

### 1. Clone the repository

```powershell
git clone https://github.com/linafourgi-tech/SWE015-Customer-Support-Ticket-Router.git
cd SWE015-Customer-Support-Ticket-Router
```

Run all of the following commands from this project root folder.

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow scripts for the current window only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install the requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `pip` installs a CUDA build of PyTorch you do not need, you can install the CPU build first: `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu`.

### 4. Download the artifact archive

```powershell
$ProgressPreference = 'SilentlyContinue'   # makes Invoke-WebRequest much faster in Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/linafourgi-tech/SWE015-Customer-Support-Ticket-Router/releases/download/v1.0.0/support-ticket-router-artifacts.zip" -OutFile "support-ticket-router-artifacts.zip"
```

You can also download it in a browser from the release page and move it into the project root.

### 5. Verify the SHA-256 checksum

```powershell
(Get-FileHash .\support-ticket-router-artifacts.zip -Algorithm SHA256).Hash -eq "f4779dc3eac1c8c8775c8534593957740a646e9d3bd0ebc240ba14a4b246a210"
```

This must print `True`. If it prints `False`, the download is incomplete or corrupted: delete the file and download it again. (`Get-FileHash` shows the hash in upper case; the `-eq` comparison ignores case.)

### 6. Extract the archive into the project root

Every path inside the ZIP already starts with `artifacts/`, so extract it **into the project root** (not into an `artifacts` folder):

```powershell
Expand-Archive -Path .\support-ticket-router-artifacts.zip -DestinationPath .
Test-Path .\artifacts\student_int8\model_int8.pt
```

`Test-Path` must print `True`. The archive restores `artifacts/labels.json`, `artifacts/classical/`, `artifacts/mlp/`, `artifacts/bert/`, `artifacts/student/` and `artifacts/student_int8/`. It does not contain `artifacts/results.json`, which is already in the repository, so nothing committed is overwritten. After extracting, you can delete the ZIP. Git ignores both the ZIP and the extracted files.

### 7. Run the app

```powershell
streamlit run app.py
```

Streamlit prints a local URL (normally http://localhost:8501) and opens it in the browser. Loading the four models takes a few seconds on the first request. To keep the app reachable only from your own computer, use `streamlit run app.py --server.address localhost`.

**Offline use:** once the artifacts are extracted, the app needs no internet connection. All models, tokenizers and labels are loaded from the local `artifacts/` folder.

## Train and benchmark (one command)

```powershell
python -m src.train
```

Defaults: `--seed 42 --epochs 30 (MLP) --bert-epochs 2 --student-layers 4 --student-epochs 3 --max-length 96`.

On the development laptop (Intel Core i7-1165G7, 4 cores / 8 threads, 11.8 GB RAM, no GPU), the full training run took about 1 h 45 min (estimated from log and artifact file timestamps). The committed `artifacts/results.json` comes from the later `--evaluate-only` run, so its `wall_clock_minutes_this_run` field records only that evaluation run (9.65 min of evaluation), not the timings of the training stages.

The script:

1. Downloads and normalizes CLINC150 `plus` (train/validation/test).
2. Trains TF-IDF + logistic regression.
3. Trains skip-gram Word2Vec on the training texts, then the MLP (SGD with momentum 0.9, lr 0.05, cross-entropy, best epoch by validation loss).
4. Fine-tunes `bert-base-uncased` (AdamW, lr 2e-5, batch 32, best epoch by validation loss).
5. Builds a 4-layer student from the fine-tuned teacher and trains it with knowledge distillation (see below).
6. Applies dynamic INT8 quantization to the distilled student, saves it, **reloads it from disk** and checks that it contains INT8 Linear layers.
7. Reloads every saved model from `artifacts/` one at a time (as the app does), evaluates it on the 5,500-example test split, frees it, and writes `artifacts/results.json`.

The test split is used only in step 7. Model selection (best epoch) uses the validation split only.

To re-run only the benchmark on saved models (for example after downloading the release archive, or when training finished but evaluation was interrupted by low memory):

```powershell
python -m src.train --evaluate-only
```

This uses the same test split and `benchmark()` code. `results.json["run_mode"]` records which mode produced the file, and `training_config` records the training settings (from `artifacts/training_config.json` when it exists).

Requirements and warnings for `--evaluate-only`:

- **Internet is required.** The script downloads CLINC150 `plus` from Hugging Face to get the test split and to check that `artifacts/labels.json` matches the dataset.
- **The complete artifact archive is required**, including `artifacts/student/` (the FP32 student). The benchmark always evaluates this supplementary model and stops with an error if the folder is missing. The app alone does not need `artifacts/student/`.
- **It overwrites `artifacts/results.json`.** Your new latency values will differ from the committed ones, because latency depends on the machine and its load. To keep the committed file, copy it first (e.g. `Copy-Item artifacts\results.json results_original.json`) or restore it afterwards with `git checkout -- artifacts/results.json`.

## Run the application

```powershell
streamlit run app.py
```

The app needs the artifacts from the release archive (see Quick start) or from the training command. Once they are in place, it runs offline. It accepts a new customer message and:

- normalizes the text with regular expressions and extracts a product model code (e.g. `WM-3500` → `WM3500`);
- corrects one-edit typos in product names using Levenshtein edit distance (e.g. `dishwaser` → `dishwasher`);
- replies to greetings/thanks with rule-based small talk, plus ELIZA-style reflections such as "I feel my oven is not heating" → "Why do you feel your oven is not heating?";
- predicts the intent with the distilled INT8 student and shows its confidence;
- routes out-of-scope (`oos`) or low-confidence predictions (sidebar threshold, default 0.45) to a **human agent**, and otherwise maps the intent to a team;
- optionally shows all four models' predictions side by side.

A message that contains only a greeting or thanks is answered automatically. A greeting followed by a request (e.g. "hi, my card was declined") is still classified and routed.

## Artifacts (published as a release asset, not committed)

| Path | Contents |
|---|---|
| `artifacts/labels.json` | 151 label names; index = model output id |
| `artifacts/classical/model.joblib` | TF-IDF + logistic regression pipeline |
| `artifacts/mlp/vectors.kv*`, `artifacts/mlp/model.pt` | Word2Vec vectors and MLP weights |
| `artifacts/bert/` | Fine-tuned 12-layer teacher (FP32 safetensors, config, tokenizer) |
| `artifacts/student/` | Distilled 4-layer student before quantization (FP32) |
| `artifacts/student_int8/` | `model_int8.pt` (INT8 state_dict), `config.json`, tokenizer: **loaded by the app** |
| `artifacts/results.json` | All metrics, hardware, versions, timings, example errors |
| `artifacts/train.log` | Training log (if you redirect output there) |

`artifacts/` is in `.gitignore`: model weights, logs and all other artifacts are not committed to Git. The only exception is `artifacts/results.json` (the measured benchmark, no weights), which is committed so that the reported numbers can be checked.

There are two ways to obtain the other artifacts:

1. **Download** `support-ticket-router-artifacts.zip` from the [v1.0.0 release](https://github.com/linafourgi-tech/SWE015-Customer-Support-Ticket-Router/releases/tag/v1.0.0), verify its SHA-256 and extract it into the project root (Quick start steps 4–6). It contains 22 files: `labels.json` and the `classical/`, `mlp/`, `bert/`, `student/` and `student_int8/` folders listed above. These are the exact models behind the measured results. It contains no logs and no `results.json`.
2. **Retrain** with `python -m src.train`. This takes hours on a CPU, and the new models and results will differ slightly from the published ones.

## Dataset

CLINC150 (Larson et al., 2019), `plus` configuration, loaded from Hugging Face (`clinc_oos`, `plus`):

- 150 in-scope intents across 10 domains (banking, credit cards, travel, work, utility, small talk, etc.) + 1 out-of-scope class `oos` = 151 labels
- Train 15,250 / validation 3,100 / test 5,500 (official splits; the test split includes 1,000 `oos` examples)

CLINC150 is a general virtual-assistant intent dataset. It is used here as a stand-in for support-ticket intents. It does **not** contain appliance fault types, so the product-name and model-code features in the app are rule-based extras whose accuracy is not measured by the benchmark.

Preprocessing (`src/data.py: normalize`): lowercase, merge product model codes (`wm-3500`, `wm3500`, and space-separated `WM 3500` only when typed in upper case), collapse characters repeated 3+ times to 2, remove punctuation except `'` and `-`, collapse whitespace. The same function is used in training, evaluation and the app.

## Knowledge distillation

- Teacher: `bert-base-uncased` fine-tuned on CLINC150 (12 layers, hidden size 768, ~109.6M parameters), frozen during distillation (`eval()` mode, `requires_grad=False`, `torch.no_grad()`).
- Student: same config with `num_hidden_layers=4` (~52.9M parameters; about 23.4M of these are the token embedding table, which is the same size as the teacher's). Initialized from the teacher's embeddings, its first 4 encoder layers, pooler and classifier.
- Loss: `0.5 · T² · KL(softmax(teacher/T) ‖ softmax(student/T)) + 0.5 · CrossEntropy(student, label)`, with T = 2.0.
- AdamW, lr 3e-5, weight decay 0.01, batch 32, 3 epochs, gradient clipping 1.0. The epoch with the lowest validation distillation loss is kept.
- Tokenizer, `label2id` and `id2label` are the teacher's, so output ids match `labels.json`.

## INT8 quantization

- `torch.ao.quantization.quantize_dynamic(student, {nn.Linear}, dtype=torch.qint8)`, applied to the **distilled student**, not the teacher.
- All 26 Linear layers become INT8 (4 layers × 6 Linear layers + pooler + classifier). Weights are stored as int8 and activations are quantized on the fly at inference time. Embeddings and LayerNorm stay FP32, which is why the INT8 file is not 4× smaller.
- Saved as a state_dict (`model_int8.pt`) plus `config.json`. `src/models.py: load_int8_student` rebuilds the student from the config, re-applies the same quantization, and loads the int8 weights. A round-trip test confirmed identical logits after reloading. On this machine PyTorch uses the `onednn` quantized engine.

## Benchmark methodology

- **Accuracy** on all 5,500 test examples; **OOS recall** = recall of the `oos` class.
- **Latency**: CPU, batch size 1, one string per call, including `normalize()`, tokenization/feature extraction, forward pass and softmax. 10 warm-up calls, then 100 timed calls (`time.perf_counter`) over the first 100 test messages. Mean and standard deviation are reported in ms. PyTorch uses its default thread count (recorded in the JSON).
- **Size (MB)**: on-disk size of the files needed to run each model. TF-IDF: the joblib file. MLP: Word2Vec vectors + MLP weights. BERT teacher / students: weights + `config.json` + tokenizer files. The exact definition is stored per model in `size_includes`.
- Hardware and software versions are written to `results.json["hardware"]`.
- A supplementary row `bert_student_fp32` (not part of the four-model ladder) separates the effect of distillation from the effect of quantization.

Measurements are machine-specific. The committed `artifacts/results.json` contains the values measured on the development laptop (below). If you re-run training or `--evaluate-only`, report the values from your own `artifacts/results.json`.

### Measured results (development laptop)

From `artifacts/results.json`. Models trained on 2026-11-01 with the default settings; benchmark produced by `python -m src.train --evaluate-only` on the saved models after the training run's own evaluation was stopped for low memory.

| Model | Accuracy | OOS recall | Size (MB) | Mean latency ± std (ms), 100 runs |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.8229 | 0.470 | 14.3 | 1.33 ± 1.13 |
| Word2Vec + MLP | 0.6356 | 0.258 | 2.4 | 0.21 ± 0.11 |
| BERT teacher (12 layers, FP32) | 0.8427 | 0.420 | 419.0 | 56.76 ± 9.58 |
| **Distilled student (4 layers, INT8), used by the app** | **0.8385** | **0.460** | **119.7** | **14.76 ± 6.95** |
| Supplementary: distilled student (FP32) | 0.8467 | 0.496 | 202.7 | 18.93 ± 2.90 |

Hardware: Intel Core i7-1165G7 @ 2.80 GHz (4 cores / 8 threads), 11.8 GB RAM, no GPU, Windows 11, Python 3.12.10, PyTorch 2.9.1+cpu (4 threads, `onednn` quantized engine).

## Project structure

- `src/data.py`: dataset loading, normalization, model-code extraction
- `src/models.py`: MLP, mean-embedding features, INT8 quantization/loading helpers
- `src/train.py`: trains, distills, quantizes and benchmarks all models
- `app.py`: Streamlit application
- `PROJECT_MAP.md`: requirement-to-code map and architecture diagram
- `PRESENTATION.md`: slide content for the presentation (filled with the measured results)
- `presentation/SWE015_Support_Ticket_Router_Presentation.pdf`: final 12-slide deck, generated from `presentation/slides.html` (Chrome/Edge → Print → Save as PDF)

## Known limitations

- CLINC150 intents are general assistant intents, not real support tickets. The intent-to-team mapping in `app.py` is a hand-written prefix mapping and is not evaluated (e.g. `damaged_card` and `transfer` fall through to "General support").
- OOS recall is measured with plain argmax. The app's confidence threshold sends more messages to a human agent, but this thresholded behavior is not measured by the benchmark. Confident mistakes (e.g. "how do i freeze my credit card" → `damaged_card`, 66%) still pass the threshold.
- The teacher was fine-tuned for only 2 epochs on CPU, and its validation loss was still decreasing (2.659 → 1.025).
- Product extraction covers a fixed list of 7 products, with one-edit typo correction for words of 5+ letters. It can over-correct (e.g. "my machines are broken" becomes "my machine are broken" and is tagged as a washing machine) and is not evaluated.
- Space-separated lower-case model codes ("wm 3500") are intentionally not merged, to avoid gluing ordinary words to numbers ("send 100").
- Dynamic quantization does not quantize the embedding table, so most of the remaining student size is FP32 embeddings.
- Latency depends on the machine, background load and thread count; it was measured on a laptop, not a server.
- `torch.ao.quantization` is deprecated; a future migration to `torchao` will be needed for PyTorch ≥ 2.10.
- CLINC150 itself contains a few exact duplicate texts across train and test (17 after normalization). This comes from the dataset, not from data leakage in this code.

## Data and model citations

- Larson et al. (2019), [*An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*](https://aclanthology.org/D19-1131/), EMNLP-IJCNLP.
- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL-HLT.
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*.

Check dataset and model licenses before redistribution. Do not commit downloaded datasets, model weights, API tokens, or local caches.

## AI use disclosure

The presentation must contain an honest AI usage statement. List each AI tool that was actually used and what it helped with, for example: project planning, initial code drafting, auditing and fixing the distillation/quantization pipeline, writing tests, and drafting documentation. Also say what you reviewed, changed, ran and verified yourself. See slide 12 of the presentation (`presentation/slides.html` / `PRESENTATION.md`), and edit it so it matches what actually happened.
