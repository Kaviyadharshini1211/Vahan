# ASR Shootout — Benchmarking Indian Conversational Speech for Locality Extraction

## Overview

This project benchmarks multiple Automatic Speech Recognition (ASR) systems for Indian conversational speech in noisy, real-world telephony-like conditions.

The target use case is a blue-collar hiring platform where candidates interact via phone calls and WhatsApp voice notes, often using Hinglish or regional speech patterns while mentioning locality names.

The primary objective of this evaluation was not just transcription quality, but:

- Locality/entity recognition
- Robustness to noisy conversational speech
- Latency and deployability tradeoffs
- Performance under code-switching and rushed speech

Deepgram was used as the required baseline, and compared against open-source Whisper-based systems.

---

# Dataset Creation

## Self-Recorded Dataset

I recorded conversational audio samples containing Bangalore locality names under realistic speaking conditions.

Instead of reading locality names directly, I embedded them inside natural conversational sentences such as:

> “Haan main Koramangala mein rehta hoon”

The goal was to simulate realistic hiring-platform conversations rather than studio-quality speech.

## Recording Conditions

The dataset intentionally included varied acoustic and conversational conditions:

| Condition | Examples |
|---|---|
| Quiet room | Indoor clean recordings |
| Street noise | Traffic/background noise |
| Phone-call style | Compressed mobile-call speech |
| Whispered | Low-volume utterances |
| Rushed speech | Fast conversational delivery |
| Code-switched | Hindi + English mixed speech |

This variation was important because production ASR systems fail differently depending on recording conditions.

---

# Models Evaluated

## 1. Deepgram (Baseline)

Deepgram was evaluated as the required API baseline.

### Why chosen
- Fast cloud API
- Streaming support
- Production-ready deployment
- Commonly used commercial ASR provider

### Limitations observed
- Weak recognition of Indian locality names
- Poor entity extraction for Hinglish speech
- Lower robustness to code-switching

---

## 2. OpenAI Whisper

Whisper Large was evaluated as a high-accuracy open-source multilingual ASR model.

### Why chosen
- Strong multilingual support
- Widely adopted open-source benchmark model
- Known robustness to noisy speech

### Limitations observed
- Higher latency
- Larger memory footprint
- Hallucinations in difficult/noisy audio

---

## 3. Faster-Whisper

Faster-Whisper was evaluated as an optimized Whisper implementation.

### Why chosen
- Lower inference latency
- Better deployment efficiency
- Offline inference capability
- Lower operational cost than API-based systems

### Key observation
Faster-Whisper achieved similar entity recognition quality to Whisper Large while reducing inference latency significantly.

---

# Evaluation Metrics

Traditional ASR benchmarks often rely only on Word Error Rate (WER), but locality extraction systems require more task-specific evaluation.

The following metrics were used:

| Metric | Purpose |
|---|---|
| WER | Measures transcription word accuracy |
| CER | Character-level transcription accuracy |
| Entity Hit Rate | Whether locality name was captured correctly |
| Fuzzy Entity Match | Phonetic/approximate locality matching |
| Latency | End-to-end inference speed |

---

# Important Finding: WER Was Insufficient

One of the strongest findings from this benchmark was that WER significantly underestimates ASR quality for Hinglish conversational speech.

Example:

| Reference | Prediction |
|---|---|
| Koramangala | Kora Mangala |

Semantically, the prediction is understandable and usable for downstream entity extraction.

However, WER heavily penalizes this prediction because the spelling differs.

This became especially common for:

- Indian locality names
- Code-switched Hinglish
- Transliteration inconsistencies
- Phonetic spelling variation

As a result, entity-level metrics proved more useful than raw WER for this task.

---

# Benchmark Results

## Overall Comparison

| Model | Avg WER | Entity Hit | Avg Latency |
|---|---|---|---|
| Deepgram | Lower WER but poor entity capture | 0.00 | Fastest |
| Whisper Large | Better locality understanding | 0.35 | Slowest |
| Faster-Whisper | Similar accuracy to Whisper | 0.35 | Faster than Whisper |

## Key Observation

Although Deepgram often produced lower WER values, it consistently struggled to correctly capture locality entities.

Open-source Whisper-based models performed significantly better for locality extraction and Hinglish conversational speech.

---

# Condition-wise Analysis

## Hardest Audio Conditions

| Condition | Observation |
|---|---|
| Rushed speech | Highest failure rate |
| Phone-call audio | Significant degradation |
| Street noise | Moderate degradation |
| Whispered speech | Surprisingly robust |

## Most Important Insight

Rushed conversational speech degraded performance more than background noise.

This was unexpected.

Initially, I expected street-noise recordings to be the hardest condition. However, fast conversational delivery caused significantly larger transcription degradation across all models.

This suggests conversational timing and articulation variability may be more damaging than environmental noise for locality extraction.

---

# Failure Analysis

## 1. Phonetic Locality Splitting

Examples:

| Ground Truth | Model Output |
|---|---|
| Koramangala | Kora Mangala |
| Thanisandra | Sani Sandra |

These outputs were often semantically understandable to humans but penalized heavily by WER.

---

## 2. Hallucinations in Smaller Models

Smaller Faster-Whisper models occasionally produced repetitive or nonsensical outputs under noisy conditions.

Example:

> “ॐ ॐ ौ ौ ौ ौ...”

This occurred most frequently during:

- phone-call recordings
- whispered speech
- low-volume noisy clips

Larger models were more stable and hallucinated less frequently.

---

## 3. Code-Switched Speech

Deepgram struggled significantly with Hindi-English mixed speech compared to Whisper-based models.

Whisper and Faster-Whisper handled:

- Hindi locality names
- English words
- mixed conversational structure

more reliably.

---

# Model Size Tradeoff Experiment

I additionally compared Faster-Whisper model sizes:

- small
- medium
- large-v3

## Observation

Unexpectedly, large-v3 achieved both:

- the best locality recognition
- and lower latency than smaller variants on my setup

Possible reasons:

- fewer decoding repetitions
- more stable beam-search behavior
- reduced hallucination loops

This highlights that smaller models are not always faster in real conversational ASR workloads.

---

# Production Considerations

## Deepgram

### Advantages
- Lowest latency
- Managed cloud infrastructure
- Easy deployment
- Streaming support

### Disadvantages
- API cost at scale
- Poor locality extraction quality
- Dependent on internet connectivity

---

## Whisper Large

### Advantages
- Strong multilingual accuracy
- Better entity understanding

### Disadvantages
- High memory usage
- Slow inference
- More difficult deployment

---

## Faster-Whisper

### Advantages
- Offline inference
- Lower latency
- Strong locality extraction
- Better deployment efficiency
- No API dependency

### Disadvantages
- Requires local compute
- GPU memory constraints for larger models

---

# Final Recommendation

For Indian conversational telephony-style ASR workloads focused on locality extraction, Faster-Whisper large-v3 provided the best overall tradeoff between:

- locality recognition quality
- latency
- deployment practicality
- multilingual robustness

Deepgram remained the fastest option, but struggled with Hinglish locality extraction.

Traditional WER alone proved insufficient for evaluating production ASR quality in this setting.

Entity-level evaluation and qualitative failure analysis provided much more useful insight.

---

# Repository Contents

The repository contains:

- Benchmark pipeline
- Audio recordings
- Result CSV/JSON files
- Charts and visualizations
- Condition-wise analysis
- Model-size experiments
- Reproducible evaluation code

---

# Key Takeaways

1. WER significantly underestimates ASR usefulness for Hinglish locality recognition.
2. Entity-level evaluation is more meaningful for production telephony systems.
3. Rushed conversational speech was more damaging than environmental noise.
4. Whisper-based open-source models outperformed the commercial baseline for locality extraction.
5. Faster-Whisper large-v3 offered the best production tradeoff overall.
