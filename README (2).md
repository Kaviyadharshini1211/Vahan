# ASR Shootout — Benchmarking Indian Conversational Speech

Benchmarking Automatic Speech Recognition (ASR) systems for Indian conversational telephony-style speech with a focus on locality extraction.

This project evaluates multiple ASR systems under noisy, real-world conditions using self-recorded Bangalore locality audio samples.

---

# Objective

The goal of this benchmark is to evaluate how well modern ASR systems handle:

- Hinglish conversational speech
- Indian locality names
- Noisy phone-call conditions
- Code-switched utterances
- Rushed and whispered speech

The benchmark compares Deepgram (baseline) against Whisper-based open-source systems.

---

# Models Evaluated

| Model | Type |
|---|---|
| Deepgram | Cloud API baseline |
| Whisper Large | Open-source multilingual ASR |
| Faster-Whisper | Optimized Whisper inference |

---

# Dataset

## Self-recorded conversational dataset

The dataset consists of conversational utterances containing Bangalore locality names such as:

- Koramangala
- Indiranagar
- Whitefield
- Electronic City
- Hebbal
- Thanisandra

Recordings were intentionally captured under varied real-world conditions:

- quiet-room
- street-noise
- phone-call
- rushed-speech
- whispered
- code-switched Hinglish

---

# Repository Structure

```text
asr-shootout/
│
├── README.md
├── report.md
├── requirements.txt
├── asr_benchmark.py
├── notebook.ipynb
│
├── recordings/
│   ├── 01_koramangala_quiet.wav
│   ├── ...
│
├── results/
│   ├── results.csv
│   ├── results.json
│   ├── model_size_experiment.csv
│   ├── condition_analysis.csv
│
├── charts/
│   ├── wer_by_model.png
│   ├── entity_hit_by_model.png
│   ├── latency_vs_accuracy.png
```

---

# Evaluation Metrics

The following metrics were used:

| Metric | Description |
|---|---|
| WER | Word Error Rate |
| CER | Character Error Rate |
| Entity Hit | Exact locality extraction |
| Fuzzy Entity Match | Approximate locality match |
| Latency | End-to-end inference latency |

---

# Key Findings

## 1. WER alone was insufficient

Traditional WER heavily penalized phonetically correct Hinglish outputs.

Example:

| Ground Truth | Prediction |
|---|---|
| Koramangala | Kora Mangala |

Semantically understandable predictions were still treated as incorrect by WER.

Entity-level metrics proved more meaningful for this use case.

---

## 2. Faster-Whisper provided the best tradeoff

Faster-Whisper large-v3 achieved:

- strong locality recognition
- lower latency than Whisper Large
- offline deployability
- no API dependency

---

## 3. Rushed speech was the hardest condition

Unexpectedly, rushed conversational speech degraded ASR quality more than background noise.

This suggests articulation variability may be more damaging than environmental noise for locality extraction tasks.

---

# Installation

## Clone repository

```bash
git clone <repo-url>
cd asr-shootout
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Benchmark

## Configure API keys

Set environment variables:

```bash
export DEEPGRAM_API_KEY=<your-key>
export SARVAM_API_KEY=<your-key>
```

---

## Run benchmark

```bash
python asr_benchmark.py \
  --audio-dir recordings \
  --models deepgram,whisper,faster_whisper \
  --whisper-size large \
  --output results/results.json \
  --charts
```

---

# Output Files

The benchmark generates:

| File | Purpose |
|---|---|
| results.csv | Raw benchmark outputs |
| results.json | Structured benchmark results |
| charts/ | Visualizations |
| model_size_experiment.csv | Model-size tradeoff analysis |
| condition_analysis.csv | Condition-wise evaluation |

---

# Charts Generated

The benchmark generates:

- WER by model
- Entity hit rate by model
- Latency vs accuracy

---

# Production Recommendation

For Indian conversational telephony-style ASR focused on locality extraction:

## Recommended model

### Faster-Whisper large-v3

Reasoning:

- strong multilingual robustness
- better locality recognition
- lower latency than Whisper Large
- deployable offline
- lower operational cost than API-based systems

---

# Limitations

- Small self-recorded dataset
- Limited speaker diversity
- WER not ideal for Hinglish transliteration
- No streaming ASR evaluation

---

# Future Improvements

Potential future extensions:

- larger multilingual datasets
- streaming ASR benchmarking
- speaker-diarization evaluation
- semantic similarity metrics
- entity-aware decoding

---

# Author

ASR Shootout Internship Assignment Submission
