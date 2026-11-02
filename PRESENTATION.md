# Presentation: SWE015 Use Case 2, Customer Support Ticket Router

The final slide deck is `presentation/SWE015_Support_Ticket_Router_Presentation.pdf`, 12 slides, 16:9. It is generated from `presentation/slides.html`. To regenerate it, open the HTML file in Chrome or Edge and choose Print → Save as PDF, with margins "None" and background graphics on.

This file holds the same content as text, for speaker notes and review. It follows the required order: **Introduction → Development → Conclusion**. Every number comes from `artifacts/results.json` (evaluate-only run on the models trained on 2026-11-01) or from observed app outputs. Fill in the items in `[brackets]` before submitting.

---

## INTRODUCTION

### Slide 1: Title
- **Customer Support Ticket Router**: intent routing with a distilled, INT8-quantized BERT running on a CPU laptop
- SWE015 Introduction to Large Language Models, Homework: Building an Intelligent Application, Use Case 2
- Student: [Your name] · Student ID: [Student ID] · Repository: [Repository URL] · Demo video: [Demo video URL]

### Slide 2: Problem and goal
- Support teams receive free-text messages that must reach the right queue quickly.
- **Goal:** classify each message into an intent, map it to a team, and send uncertain or out-of-scope messages to a human agent.
- **Constraint:** everything runs locally on a CPU-only laptop, so the model in the app must be small and fast.
- **Approach:** a four-model ladder from classical NLP to fine-tuned BERT; BERT is compressed by knowledge distillation followed by INT8 quantization.
- Success criteria, all on one held-out test set: accuracy, OOS recall, size (MB), CPU latency.

---

## DEVELOPMENT

### Slide 3: Dataset and preprocessing
- **CLINC150 `plus`** (Larson et al., 2019), from Hugging Face `clinc_oos`: 150 intents in 10 domains + `oos` = **151 labels**.
- Splits: train 15,250 (250 oos) · validation 3,100 (100 oos, used for epoch selection) · test 5,500 (1,000 oos, final evaluation only).
- **Scope:** CLINC150 is a general assistant-intent dataset (banking, travel, small talk, …), **not an appliance-fault dataset**. The product and model-code features are rule-based and were not evaluated.
- Regex normalization (`src/data.py`): merge model codes before lowercasing (`WM-3500` → `wm3500`); lowercase; `loooove` → `loove`; strip punctuation; collapse whitespace. Training, evaluation and the app all use this same function.
- Bug fixed in the audit: the original regex also joined ordinary words to numbers (`send 100` → `send100`) in 646 of 23,850 texts (2.7%).

### Slide 4: System architecture
The PDF has a rendered diagram with two lanes:
- **Offline training** (`python -m src.train`, CPU): CLINC150 → `normalize()` → TF-IDF + LogReg | Word2Vec + MLP (SGD) | BERT teacher (12 layers, FP32) → knowledge distillation (4-layer student, KL with T = 2 + CE) → dynamic INT8 (26 Linear layers). All four ladder models → evaluation on the same 5,500 test messages → `results.json`.
- **Application** (`streamlit run app.py`, local CPU): customer message → product-name correction (edit distance + model-code regex) → `normalize()` → distilled INT8 student (the only routing model; loaded from `artifacts/student_int8`) → "intent = oos or confidence < 0.45?" → yes: Human agent / no: team from the intent-prefix mapping. The other three models produce a side-by-side comparison only. In parallel, ELIZA-style rules on the raw message: greeting/thanks only → small-talk reply, no ticket routed; reply plus a request → the request is still routed.

### Slide 5: Technique map
| Area | Technique | Code |
|---|---|---|
| A · Neural networks | Skip-gram Word2Vec (100-d), mean-pooled → MLP 100→128→151 (ReLU, dropout), SGD + momentum, backpropagation | `train.py: train_mlp`, `models.py: MeanEmbeddingMLP` |
| B · Classical NLP | Regex normalization + model-code extraction; Levenshtein product-name correction; ELIZA-style small talk with pronoun reflection; TF-IDF (1–2-grams) + logistic regression | `data.py`, `app.py: edit_distance, small_talk_reply` |
| C · Transformers | Fine-tuned `bert-base-uncased` (12 layers, hidden 768, 151 labels) | `train.py: train_models` |
| D · Efficient models | Knowledge distillation → 4-layer student, **then** dynamic INT8 quantization; the app loads this model | `train.py: create_student_model, train_student`, `models.py: quantize_int8` |

### Slide 6: Four-model ladder and training setup
| # | Model | Training |
|---|---|---|
| 1 | TF-IDF + logistic regression | 1–2-grams, min_df 2, sublinear TF; LBFGS, C = 4 |
| 2 | Word2Vec + MLP | Skip-gram 100-d, window 5, 8 epochs; MLP: SGD lr 0.05, momentum 0.9, 30 epochs, best epoch by validation loss |
| 3 | BERT teacher (12 layers, FP32) | `bert-base-uncased`, AdamW lr 2e-5, batch 32, 2 epochs, max length 96 (validation loss 2.659 → 1.025) |
| 4 | Distilled student, 4 layers, INT8 | Initialized from the teacher, 3 epochs of distillation (validation loss 0.705 → 0.471 → 0.398), then INT8 |

- Same seed (42), splits and normalization for all models. The test split is used once.
- Training took about 1 h 45 min on the laptop CPU (teacher ≈ 54 min, distillation ≈ 51 min). These times are estimated from log and artifact file timestamps.

### Slide 7: Distillation first, then quantization
- Teacher: 12 layers, 109.6 M parameters, FP32 → distilled student: 4 layers, 52.9 M, FP32 → INT8 student (app): 26 Linear layers with int8 weights.
- Distillation (Hinton et al., 2015): the student is initialized from the teacher's embeddings, first 4 layers, pooler and classifier. The teacher is frozen. Loss = 0.5 · T² · KL(teacher ‖ student) + 0.5 · CE(label), T = 2. AdamW lr 3e-5; the best validation epoch is kept. Parameters drop 2.1× because the 23.4 M-parameter embedding table is unchanged.
- Quantization: `torch.ao.quantization.quantize_dynamic` on the student's `nn.Linear` layers (qint8). Weights are int8; activations are quantized at run time; engine `onednn`. Saved as state_dict + config. Reloading was verified to give identical logits and 26 INT8 layers. The app refuses to start if the model is not smaller than the teacher or has no INT8 layers.

### Slide 8: Benchmark results (same 5,500 test messages)
| Model | Accuracy | OOS recall | Size (MB) | Mean latency (ms) ± std |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.823 | 0.470 | 14.3 | 1.33 ± 1.13 |
| Word2Vec + MLP | 0.636 | 0.258 | 2.4 | 0.21 ± 0.11 |
| BERT teacher, 12 layers, FP32 | 0.843 | 0.420 | 419.0 | 56.76 ± 9.58 |
| **Distilled student, 4 layers, INT8 (app model)** | **0.839** | **0.460** | **119.7** | **14.76 ± 6.95** |

- Teacher → INT8 student: −0.4 accuracy points, 3.5× smaller, 3.8× faster on CPU.
- **Supplementary note** (not one of the four ladder models): the distilled student before quantization (FP32) scored accuracy 0.847, OOS recall 0.496, 202.7 MB, 18.93 ± 2.90 ms. Quantization therefore cost about 0.8 accuracy points.
- Hardware: Intel Core i7-1165G7 @ 2.80 GHz (4 cores / 8 threads), 11.8 GB RAM, CPU only, Windows 11, Python 3.12.10, PyTorch 2.9.1+cpu (4 threads), Transformers 4.57.6.
- Latency: batch size 1, raw string → normalize → tokenize → forward → softmax; 10 warm-ups + 100 timed runs. Size = files needed to run the model.

### Slide 9: Live application and real failure examples
| Input | INT8 student | Routing | Comment |
|---|---|---|---|
| "hi, my card was declined at the store" | `card_declined` 80.6% | Billing and payments | Greeting reply shown, request still routed |
| "My dishwaser WM-3500 is leaking water" | `w2` 9.2% | Human agent | Typo fixed, code `WM3500`; topic not in CLINC, so low confidence and the fallback works |
| **"how do i freeze my credit card"** | **`damaged_card` 66.4%** | General support | **Failure:** wrong but confident; above the threshold, so no human sees it |

Test-set errors (from `results.json`): "how can i up my credit score" → `credit_score` (0.58), true `improve_credit_score`; "vacation request please" → `pto_request_status` (0.49), true `pto_request`; "repeat what the weather will be like" → `weather` (0.85), dataset label `transfer` (the label itself looks wrong).

---

## CONCLUSION

### Slide 10: Conclusions
- Distillation + INT8 kept almost all of the teacher's accuracy (0.839 vs 0.843) with 3.5× less disk and 3.8× lower CPU latency.
- Quantization cost about 0.8 points (supplementary FP32 student: 0.847).
- The FP32 student beat the 2-epoch teacher, probably because it trained for 3 more epochs. This is not evidence that distillation beats teachers in general.
- TF-IDF + LogReg is a strong, cheap baseline (0.823, about 1 ms). Word2Vec + MLP is the weakest model (0.636).
- **Main weakness:** OOS recall is low (0.26–0.47 across the four ladder models). The app model detects only 46% of OOS test messages with plain argmax.

### Slide 11: Limitations and future work
Limitations:
- Low OOS recall. The app's threshold adds a human fallback, but its effect was not measured, and it cannot catch confident errors.
- The teacher trained for only 2 epochs; its validation loss was still falling.
- CLINC150 is not real ticket data. The team mapping and product rules are hand-written and not evaluated: "my machines are broken" becomes "my machine are broken" and is tagged as a washing machine.
- Latency varies with load (TF-IDF: 0.73 ms in the training run vs 1.33 ms in the evaluation run).
- Embeddings stay FP32; `torch.ao.quantization` is deprecated (removal planned in PyTorch 2.10).

Future work: tune the OOS threshold on validation and report the result; train the teacher longer and re-distill; evaluate on real tickets and train an appliance-fault classifier; quantize embeddings or move to `torchao`; try 2–3-layer students.

### Slide 12: AI usage statement
| Tool | How it was used |
|---|---|
| ChatGPT (OpenAI) | Project planning and drafting of the initial code, as recorded in the original project README. |
| Claude Code (Anthropic) | Auditing the repository; fixing the model-code regex bug; completing the distillation → INT8 pipeline (saving/reloading, checks, memory-safe evaluate-only mode); writing smoke and app end-to-end tests; drafting the README, project map and slides. Training and evaluation commands were launched from this tool and ran on my laptop. |
| My role | Selected Use Case 2 and set the requirements and rules for the work; reviewed the changes, outputs and results; responsible for the final submission. [Add any parts you wrote or changed yourself] |

All benchmark numbers and test outcomes were produced by running the code locally on my laptop (`artifacts/results.json`, test logs). None of them were written or invented by an AI tool.
