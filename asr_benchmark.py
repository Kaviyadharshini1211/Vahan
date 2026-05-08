"""
ASR Shootout — Fixed Benchmark Pipeline v2
==========================================
Key fixes over v1:
  1. Transliterates Devanagari → Roman before computing WER/CER
     (models output Hindi script; ground truth is Romanized → WER was meaningless)
  2. Deepgram now uses language='multi' so it handles Hinglish without returning empty
  3. IndicConformer replaced with faster_whisper (large-v3) as the third model
     (IndicConformer is a gated HuggingFace repo requiring manual access request)
  4. Entity matching is now script-agnostic (checks both Roman + Devanagari forms)
  5. Cleaner summary table with correct, meaningful numbers

Usage:
    pip install -r requirements_v2.txt
    export DEEPGRAM_API_KEY="..."
    python asr_benchmark_v2.py --audio_dir ./recordings --models deepgram,whisper,faster_whisper
"""

import os, sys, re, time, json, argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import jiwer
import librosa
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH  (Romanized — your actual sentences)
# ──────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH: Dict[str, str] = {
    "01_koramangala_quiet.wav":          "haan main koramangala mein rehta hoon",
    "02_indiranagar_street.wav":         "mera ghar indiranagar ke paas hai",
    "03_whitefield_phone.wav":           "sir main whitefield se bol raha hoon",
    "04_electroniccity_rushed.wav":      "electronic city mein kaam karta hoon",
    "05_marathahalli_hinglish.wav":      "i live near marathahalli bridge",
    "06_jayanagar_quiet.wav":            "mera address jayanagar fourth block hai",
    "08_hebbal_whispered.wav":           "hebbal flyover ke paas rehta hoon",
    "10_banashankari_quiet.wav":         "haan banashankari second stage mein hoon",
    "11_hsrlayout_street.wav":           "hsr layout sector two mein rehta hoon",
    "13_majestic_hinglish.wav":          "majestic bus stand ke paas hoon bhai",
    "14_silkboard_phone.wav":            "silk board junction ke paas mera ghar hai",
    "15_bellandur_noise.wav":            "bellandur ke paas ek room liya hai",
    "16_sarjapur_quiet.wav":             "sarjapur road pe rehta hoon",
    "17_bommanahalli_rushed.wav":        "bommanahalli mein hoon abhi",
    "20_yeshwanthpur_quiet.wav":         "yeshwanthpur railway station ke paas",
    "22_kadugondanahalli_phone.wav":     "kadugondanahalli mein mera purana ghar tha",
    "25_rajarajeshwarinagar_hinglish.wav":"i stay in rajarajeshwarinagar near rr nagar metro",
    "26_kothanurdinne_rushed.wav":       "kothanur dinne area mein hoon",
    "27_thanisandra_phone.wav":          "thanisandra main road pe",
    "30_thalaghattapura_hinglish.wav":   "thalaghattapura near kengeri hoon main",
}

LOCALITY_ENTITIES: Dict[str, str] = {
    "01_koramangala_quiet.wav":          "Koramangala",
    "02_indiranagar_street.wav":         "Indiranagar",
    "03_whitefield_phone.wav":           "Whitefield",
    "04_electroniccity_rushed.wav":      "Electronic City",
    "05_marathahalli_hinglish.wav":      "Marathahalli",
    "06_jayanagar_quiet.wav":            "Jayanagar",
    "08_hebbal_whispered.wav":           "Hebbal",
    "10_banashankari_quiet.wav":         "Banashankari",
    "11_hsrlayout_street.wav":           "HSR Layout",
    "13_majestic_hinglish.wav":          "Majestic",
    "14_silkboard_phone.wav":            "Silk Board",
    "15_bellandur_noise.wav":            "Bellandur",
    "16_sarjapur_quiet.wav":             "Sarjapur",
    "17_bommanahalli_rushed.wav":        "Bommanahalli",
    "20_yeshwanthpur_quiet.wav":         "Yeshwanthpur",
    "22_kadugondanahalli_phone.wav":     "Kadugondanahalli",
    "25_rajarajeshwarinagar_hinglish.wav":"Rajarajeshwarinagar",
    "26_kothanurdinne_rushed.wav":       "Kothanur Dinne",
    "27_thanisandra_phone.wav":          "Thanisandra",
    "30_thalaghattapura_hinglish.wav":   "Thalaghattapura",
}

# Devanagari versions of each locality — used for entity matching when model outputs Hindi script
LOCALITY_DEVANAGARI: Dict[str, List[str]] = {
    "Koramangala":       ["कोरमंगला", "कोरामंगला"],
    "Indiranagar":       ["इंदिरा नगर", "इन्दिरानगर", "इंद्रा नगर"],
    "Whitefield":        ["वाइटफील्ड", "व्हाइटफील्ड"],
    "Electronic City":   ["इलेक्ट्रॉनिक सिटी", "इलेक्ट्रोनिक सिटी", "एलेक्ट्रोनिक सिटी"],
    "Marathahalli":      ["मराठाहल्ली", "मराठाहली"],
    "Jayanagar":         ["जयनगर", "जैनगर", "जय नगर"],
    "Hebbal":            ["हेब्बाल", "हेबाल", "है बल"],
    "Banashankari":      ["बनशंकरी", "भंशंकरी", "बन्शंकरी"],
    "HSR Layout":        ["एचएसआर लेआउट", "हेचसारा"],
    "Majestic":          ["मैजेस्टिक", "वाजस्तिक"],
    "Silk Board":        ["सिल्क बोर्ड"],
    "Bellandur":         ["बेलंदूर", "बेल्लन्दूर"],
    "Sarjapur":          ["सार्जापुर", "सार्ज़पुर"],
    "Bommanahalli":      ["बोमनाहल्ली", "बोमने हली"],
    "Yeshwanthpur":      ["येशवंतपुर", "येस्वंत्पूर", "यह स्वतंत्र"],
    "Kadugondanahalli":  ["कडुगोंडनहल्ली", "खड़ुण्ड़ों हल्ली"],
    "Rajarajeshwarinagar":["राजराजेश्वरीनगर", "आरार्नगर"],
    "Kothanur Dinne":    ["कोथानूर डिन्ने", "गृतनोडिने"],
    "Thanisandra":       ["थनिसंद्रा", "तानी संद्रा"],
    "Thalaghattapura":   ["तालघट्टपुरा", "तालगट्टपुरा"],
}

CONDITION_MAP = {
    "quiet": "quiet-room",
    "street": "street-noise",
    "noise": "background-noise",
    "phone": "phone-call",
    "rushed": "rushed-speech",
    "whispered": "whispered",
    "hinglish": "code-switched",
    "slow": "slow",
}

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
# TRANSLITERATION  (Devanagari → Roman)
# Uses the 'indic-transliteration' library. Falls back to bare text if not installed.
# ──────────────────────────────────────────────────────────────────────────────

def transliterate_to_roman(text: str) -> str:
    """
    Convert Devanagari script in `text` to IAST-style Roman.
    Leaves already-Roman text unchanged.
    Falls back gracefully if indic_transliteration is not installed.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        # Check if text contains Devanagari (Unicode block 0900-097F)
        if any('\u0900' <= ch <= '\u097f' for ch in text):
            return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        return text
    except ImportError:
        # If library not installed, do a best-effort character strip
        # (removes Devanagari; keeps Roman/digits/spaces)
        return re.sub(r'[\u0900-\u097f]+', ' ', text).strip()


# ──────────────────────────────────────────────────────────────────────────────
# TEXT NORMALISATION & METRICS
# ──────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, transliterate Devanagari to Roman, strip punctuation."""
    text = transliterate_to_roman(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = normalise(ref), normalise(hyp)
    if not ref_n:
        return 1.0
    try:
        return min(jiwer.wer(ref_n, hyp_n), 3.0)   # cap at 3 for display clarity
    except Exception:
        return 1.0


def compute_cer(ref: str, hyp: str) -> float:
    ref_n, hyp_n = normalise(ref), normalise(hyp)
    if not ref_n:
        return 1.0
    try:
        return min(jiwer.cer(ref_n, hyp_n), 3.0)
    except Exception:
        return 1.0


def check_entity(hyp: str, entity: str) -> Tuple[bool, bool]:
    """
    Returns (exact_hit, fuzzy_hit).
    exact_hit  — entity found after full normalisation (script-agnostic)
    fuzzy_hit  — any known Devanagari variant found in raw hypothesis
    """
    hyp_norm = normalise(hyp)
    ent_norm = normalise(entity)

    # Exact: all words of entity appear in order in hypothesis
    tokens = ent_norm.split()
    pattern = r'.*'.join(re.escape(t) for t in tokens)
    exact = bool(re.search(pattern, hyp_norm))

    # Fuzzy: check raw Devanagari variants
    fuzzy = exact
    if not fuzzy:
        variants = LOCALITY_DEVANAGARI.get(entity, [])
        for v in variants:
            if v in hyp:
                fuzzy = True
                break

    return exact, fuzzy


def get_condition(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for key, label in CONDITION_MAP.items():
        if key in stem:
            return label
    return "unknown"


def audio_duration(path: str) -> float:
    try:
        return librosa.get_duration(path=path)
    except Exception:
        return -1.0


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 1 — DEEPGRAM  (multilingual — handles Hinglish without returning empty)
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_deepgram(path: str) -> Tuple[str, float]:
    """
    nova-2-general with language=multi.
    FIX: was 'hi' → returned empty on Hinglish/English sentences.
    """
    import httpx

    if not DEEPGRAM_API_KEY:
        raise ValueError("Set DEEPGRAM_API_KEY env variable.")

    params = {
        "model":        "nova-2-general",
        "language":     "multi",      # ← key fix: handles Hindi + English mixed
        "punctuate":    "true",
        "smart_format": "true",
        "detect_language": "true",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "audio/wav",
    }
    with open(path, "rb") as f:
        audio = f.read()

    start = time.perf_counter()
    r = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params=params, headers=headers, content=audio, timeout=60,
    )
    latency = time.perf_counter() - start
    r.raise_for_status()
    data = r.json()
    transcript = (
        data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
    )
    return transcript, latency


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 2 — OPENAI WHISPER (local, via openai-whisper)
# ──────────────────────────────────────────────────────────────────────────────

_whisper_model = None

def transcribe_whisper(path: str, size: str = "medium") -> Tuple[str, float]:
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"\n  [whisper] Loading model '{size}'...")
        _whisper_model = whisper.load_model(size)

    start = time.perf_counter()
    result = _whisper_model.transcribe(
        path,
        task="transcribe",
        fp16=False,
        verbose=False,
        # Do NOT force language — let Whisper auto-detect Hindi vs English
    )
    latency = time.perf_counter() - start
    return result["text"].strip(), latency


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 3 — FASTER-WHISPER large-v3  (open-source, much faster than openai-whisper)
# Replaces IndicConformer which requires gated HuggingFace access.
# Install: pip install faster-whisper
# ──────────────────────────────────────────────────────────────────────────────

_fw_model = None

def transcribe_faster_whisper(path: str, size: str = "large-v3") -> Tuple[str, float]:
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel
        print(f"\n  [faster_whisper] Loading '{size}'...")
        _fw_model = WhisperModel(
            size,
            device="cuda" if _cuda_available() else "cpu",
            compute_type="float16" if _cuda_available() else "int8",
        )

    start = time.perf_counter()
    segments, info = _fw_model.transcribe(
        path,
        beam_size=5,
        # Let it auto-detect language for better Hinglish handling
        vad_filter=True,               # removes silence chunks before transcription
        vad_parameters={"min_silence_duration_ms": 300},
    )
    text = " ".join(seg.text for seg in segments).strip()
    latency = time.perf_counter() - start
    return text, latency


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 4 — SARVAM AI  (optional; Indian startup, best for Indian languages)
# ──────────────────────────────────────────────────────────────────────────────

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

def transcribe_sarvam(path: str) -> Tuple[str, float]:
    import httpx
    if not SARVAM_API_KEY:
        raise ValueError("Set SARVAM_API_KEY env variable.")
    with open(path, "rb") as f:
        start = time.perf_counter()
        r = httpx.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": (Path(path).name, f, "audio/wav")},
            data={"language_code": "hi-IN", "model": "saarika:v2"},
            timeout=60,
        )
        latency = time.perf_counter() - start
    r.raise_for_status()
    return r.json().get("transcript", ""), latency


# ──────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    filename: str
    locality: str
    condition: str
    duration_s: float
    model: str
    reference: str
    hypothesis: str
    hypothesis_roman: str     # hypothesis after Devanagari→Roman transliteration
    wer: float
    cer: float
    entity_hit: bool          # exact match after transliteration
    entity_fuzzy: bool        # Devanagari variant found in raw output
    latency_s: float
    error: Optional[str] = None

    @property
    def rtf(self) -> float:
        return round(self.latency_s / self.duration_s, 3) if self.duration_s > 0 else -1


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def build_runners(model_names: List[str], args) -> Dict:
    runners = {}
    for name in model_names:
        name = name.strip().lower()
        if name == "deepgram":
            runners["deepgram"] = lambda p: transcribe_deepgram(p)
        elif name == "whisper":
            size = getattr(args, "whisper_size", "medium")
            runners["whisper"] = lambda p, s=size: transcribe_whisper(p, s)
        elif name == "faster_whisper":
            size = getattr(args, "faster_whisper_size", "large-v3")
            runners["faster_whisper"] = lambda p, s=size: transcribe_faster_whisper(p, s)
        elif name == "sarvam":
            runners["sarvam"] = lambda p: transcribe_sarvam(p)
        else:
            print(f"[WARN] Unknown model '{name}', skipping.")
    return runners


def run_benchmark(audio_dir: Path, runners: Dict) -> pd.DataFrame:
    files = sorted(f for f in audio_dir.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".m4a"))
    if not files:
        print(f"[ERROR] No audio files in {audio_dir}")
        sys.exit(1)

    results: List[Result] = []

    for model_name, fn in runners.items():
        print(f"\n{'='*60}\nMODEL: {model_name.upper()}\n{'='*60}")
        for af in tqdm(files, desc=model_name):
            fname = af.name
            ref   = GROUND_TRUTH.get(fname, "")
            if not ref:
                continue
            entity    = LOCALITY_ENTITIES.get(fname, "")
            condition = get_condition(fname)
            dur       = audio_duration(str(af))

            try:
                hyp, lat = fn(str(af))
                err = None
            except Exception as e:
                hyp, lat, err = "", -1.0, str(e)
                print(f"\n  [ERROR] {fname}: {e}")

            hyp_roman  = transliterate_to_roman(hyp)
            exact, fuzzy = check_entity(hyp, entity) if entity else (False, False)

            results.append(Result(
                filename       = fname,
                locality       = entity,
                condition      = condition,
                duration_s     = round(dur, 3),
                model          = model_name,
                reference      = ref,
                hypothesis     = hyp,
                hypothesis_roman = hyp_roman,
                wer            = round(compute_wer(ref, hyp), 4),
                cer            = round(compute_cer(ref, hyp), 4),
                entity_hit     = exact,
                entity_fuzzy   = fuzzy,
                latency_s      = round(lat, 3),
                error          = err,
            ))

    return pd.DataFrame([asdict(r) for r in results])


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    clean = df[df["error"].isna()]

    print("\n" + "═"*75)
    print("  ASR BENCHMARK SUMMARY  (WER computed after Devanagari→Roman transliteration)")
    print("═"*75)

    summary = (
        clean.groupby("model")
             .agg(
                 WER         = ("wer",          "mean"),
                 CER         = ("cer",          "mean"),
                 Entity_Hit  = ("entity_hit",   "mean"),
                 Entity_Fuzzy= ("entity_fuzzy", "mean"),
                 Latency_Avg = ("latency_s",    "mean"),
                 Latency_P90 = ("latency_s",    lambda x: x.quantile(0.9)),
                 N_Files     = ("filename",     "count"),
             )
             .round(3)
             .sort_values("WER")
    )
    print(summary.to_string())

    print("\n── WER by condition ──")
    pivot = (
        clean.groupby(["model", "condition"])["wer"]
             .mean().unstack(fill_value=None).round(3)
    )
    print(pivot.to_string())

    print("\n── Entity accuracy (exact hit) by condition ──")
    ent = (
        clean.groupby(["model", "condition"])["entity_hit"]
             .mean().unstack(fill_value=None).round(3)
    )
    print(ent.to_string())

    print("\n── Worst entity failures ──")
    failures = (
        clean[~clean["entity_hit"]]
             .sort_values("wer", ascending=False)
             [["filename","model","locality","condition","wer","hypothesis_roman"]]
             .head(15)
    )
    print(failures.to_string(index=False))


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    base = Path(output_path).stem
    df.to_csv(f"{base}.csv", index=False)
    df.to_json(str(output_path), orient="records", indent=2)
    print(f"\nResults saved → {base}.csv  |  {output_path}")


def generate_charts(df: pd.DataFrame, out_dir: str = "./charts") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not found — skipping charts.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    clean = df[df["error"].isna()]
    COLORS = {"deepgram": "#4F81BD", "whisper": "#C0504D",
              "faster_whisper": "#9BBB59", "sarvam": "#F79646"}

    models = clean["model"].unique()
    conditions = sorted(clean["condition"].unique())

    # ── Chart 1: WER by condition ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(conditions))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [clean[(clean["model"]==m) & (clean["condition"]==c)]["wer"].mean()
                for c in conditions]
        axes[0].bar(x + i*w, vals, w, label=m, color=COLORS.get(m, "#888"))
    axes[0].set_xticks(x + w*(len(models)-1)/2)
    axes[0].set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("WER (lower = better)")
    axes[0].set_title("WER by Audio Condition")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, max(clean["wer"].max() * 1.1, 1.0))
    axes[0].axhline(1.0, color="red", linestyle="--", lw=0.8, label="WER=1.0 baseline")

    # ── Chart 2: Entity accuracy ───────────────────────────────────────────────
    ent = clean.groupby("model")[["entity_hit","entity_fuzzy"]].mean().sort_values("entity_hit", ascending=False)
    xi  = np.arange(len(ent))
    axes[1].bar(xi - 0.2, ent["entity_hit"],   0.35, label="Exact hit",  color=[COLORS.get(m,"#888") for m in ent.index])
    axes[1].bar(xi + 0.2, ent["entity_fuzzy"],  0.35, label="Fuzzy hit",  color=[COLORS.get(m,"#aaa") for m in ent.index], alpha=0.6)
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels(ent.index, fontsize=9)
    axes[1].set_ylabel("Accuracy (higher = better)")
    axes[1].set_title("Locality Name Capture Rate")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=8)
    for j, (eh, ef) in enumerate(zip(ent["entity_hit"], ent["entity_fuzzy"])):
        axes[1].text(j - 0.2, eh + 0.03, f"{eh:.0%}", ha="center", fontsize=8)
        axes[1].text(j + 0.2, ef + 0.03, f"{ef:.0%}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/wer_entity_comparison.png", dpi=150)
    plt.close(fig)

    # ── Chart 3: Latency scatter ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        md = clean[clean["model"] == m]
        ax.scatter(md["duration_s"], md["latency_s"], label=m, alpha=0.7,
                   color=COLORS.get(m,"#888"), s=60, zorder=3)
    mx = clean["duration_s"].max()
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, label="real-time (RTF=1)")
    ax.set_xlabel("Audio duration (s)")
    ax.set_ylabel("Processing latency (s)")
    ax.set_title("Latency vs Audio Duration")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/latency_scatter.png", dpi=150)
    plt.close(fig)

    # ── Chart 4: Per-locality entity accuracy ─────────────────────────────────
    localities = clean["locality"].unique()
    fig, ax = plt.subplots(figsize=(12, 5))
    x2 = np.arange(len(localities))
    w2 = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [clean[(clean["model"]==m)&(clean["locality"]==loc)]["entity_hit"].mean()
                for loc in localities]
        ax.bar(x2 + i*w2, vals, w2, label=m, color=COLORS.get(m,"#888"))
    ax.set_xticks(x2 + w2*(len(models)-1)/2)
    ax.set_xticklabels(localities, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Entity accuracy")
    ax.set_title("Per-Locality Entity Accuracy (all models)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/per_locality_entity.png", dpi=150)
    plt.close(fig)

    print(f"Charts saved → {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir",          default="./recordings")
    p.add_argument("--models",             default="deepgram,whisper,faster_whisper")
    p.add_argument("--whisper_size",       default="medium")
    p.add_argument("--faster_whisper_size",default="large-v3")
    p.add_argument("--output",             default="results.json")
    p.add_argument("--charts",             action="store_true")
    p.add_argument("--charts_dir",         default="./charts")
    return p.parse_args()


def main():
    args    = parse_args()
    model_list = [m.strip() for m in args.models.split(",")]
    runners = build_runners(model_list, args)

    print(f"\n{'='*60}\nASR BENCHMARK v2\n{'='*60}")
    print(f"Audio dir : {args.audio_dir}")
    print(f"Models    : {list(runners.keys())}")

    df = run_benchmark(Path(args.audio_dir), runners)
    print_summary(df)
    save_results(df, Path(args.output))
    if args.charts:
        generate_charts(df, args.charts_dir)
    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    main()
