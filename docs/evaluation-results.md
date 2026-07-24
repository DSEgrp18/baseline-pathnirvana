# Sinhala TTS — Evaluation Pipeline & Results

**Project:** Development of a Natural-Sounding Sinhala TTS System (UoM CS3501, Group 18 / P15)
**Track evaluated:** Track A — VITS from scratch (Coqui-TTS) on the PathNirvana dataset
**Harness:** [`evaluate.py`](../evaluate.py)
**Status:** ✅ complete — VITS 36k vs 72k, evaluated 2026-07-24 on Kaggle T4.

---

## 1. Purpose

Quantify how our from-scratch Sinhala VITS model improves with additional training, and
produce audio samples for a native-speaker listening test. We compare two training
checkpoints of the **same** model:

| Label | Run folder | Checkpoint | Effective training |
|-------|-----------|-----------|--------------------|
| **36k** | `vits_sinhala-July-22-2026_04+42PM` (dataset Version 1) | `checkpoint_36000.pth` | ~36k steps (first session) |
| **72k** | `vits_sinhala-July-23-2026_09+23AM` (dataset Version 2) | `checkpoint_36000.pth` | ~72k effective (warm-restart continued) |

> **Note on "effective steps":** the step counter resets to 0 on a warm *restore*, so **both**
> files are literally named `checkpoint_36000.pth` — the distinguishing feature is the run
> folder timestamp (July-22 = 36k, July-23 = 72k). Effective steps = commits × ~36k.
>
> **Comparison validity:** the two evaluation runs were verified to load different files
> (`.../July-22-2026_04+42PM/...` vs `.../July-23-2026_09+23AM/...`), so this is a genuine
> two-model comparison, not the same checkpoint twice.

---

## 2. Evaluation harness

All metrics are produced by a single reproducible script, [`evaluate.py`](../evaluate.py),
run on Kaggle (GPU T4). For each checkpoint it synthesizes a **fixed held-out test set**,
then computes objective metrics and saves every wav for human rating.

Auto-discovery finds the latest checkpoint in each attached run folder and labels them
distinctly, so multiple models compare side by side; `--tag` writes each run to its own
output folder so sequential runs never overwrite each other.

---

## 3. Test set (fixed benchmark)

16 sentences, held identical across all models, spanning 6 categories that stress different
TTS behaviours (prosody, intonation, numbers, long-form coherence, code-switching, short
utterances):

| # | Category | Sentence |
|---|----------|----------|
| 0 | statement | අද දවස ඉතාම සුන්දරයි. |
| 1 | statement | මම හෙට උදේ පාසල් යනවා. |
| 2 | statement | ශ්‍රී ලංකාව ලස්සන දිවයිනකි. |
| 3 | question | ඔයාට හෙට උදේ මෙතනට එන්න පුළුවන්ද? |
| 4 | question | මේ පොත කාගේද? |
| 5 | question | ඔබ කොහෙද යන්නේ? |
| 6 | exclaim | අනේ! මේක නම් හරිම පුදුමයි! |
| 7 | exclaim | අපොයි! මට බය හිතුණා! |
| 8 | numbers | මේ පොතේ පිටු දෙසිය පනහක් තියෙනවා. |
| 9 | numbers | අද දිනය දෙදහස් විසිහයයි. |
| 10 | long | ඉස්සර කාලයේ, එක ගමක, හොඳ සිතක් ඇති, දුප්පත් ගොවියෙක් ජීවත් වුණා. |
| 11 | long | ඔහු සෑම දිනකම උදෑසන අවදි වී, තම කුඹුරට ගොස්, දහවල් වන තුරු වෙහෙස මහන්සි වී වැඩ කළේය. |
| 12 | codeswitch | මම laptop එකෙන් email එකක් යැව්වා. |
| 13 | codeswitch | අද meeting එක online තියෙනවා. |
| 14 | short | ස්තූතියි. |
| 15 | short | සුබ උදෑසනක්. |

---

## 4. Metrics

| Metric | What it measures | Tool | Direction | Trust |
|--------|------------------|------|-----------|-------|
| **WER** | Word Error Rate — intelligibility via ASR round-trip | Whisper `large-v3` + `jiwer` | lower = better | ⚠️ low (see §7) |
| **CER** | Character Error Rate — finer-grained ASR round-trip | Whisper `large-v3` + `jiwer` | lower = better | 🟡 relative only |
| **UTMOS** | Predicted naturalness MOS (~1–5) | `torch.hub` `tarepan/SpeechMOS`, `utmos22_strong` | higher = better | 🟡 rough / cross-lingual |
| **RTF** | Real-Time Factor = synth time ÷ audio duration | wall-clock | <1 = faster than real-time | ✅ reliable |
| **Human MOS** | Native-speaker naturalness rating (1–5) | listening test (§8) | higher = better | ✅ headline metric |

**How the ASR metrics are computed:** the synthesized wav is transcribed by Whisper
(`language='si'`); both reference and hypothesis are Unicode-NFC normalized and stripped of
punctuation (incl. ZWJ) before WER/CER, so we compare words/characters, not marks.

---

## 5. Environment & reproducibility

| Item | Value |
|------|-------|
| Platform | Kaggle Notebook, GPU T4 (single GPU pinned via `CUDA_VISIBLE_DEVICES=0`) |
| Model sample rate | 22.05 kHz (VITS) |
| ASR | `openai-whisper` `large-v3` |
| UTMOS | `tarepan/SpeechMOS` `utmos22_strong` |
| Key pins | `transformers==4.53.0`, `torchvision==0.23.0`, `torchaudio==2.8.0` (match torch 2.8.0) |
| Harness commit | `bfb0747` |
| Eval date | 2026-07-24 |

**Commands (two-run method, versioned dataset):**

```bash
# Run 1 — pin dataset to Version 1 (36k), then:
rm -rf repo /kaggle/working/vits_si
git clone https://github.com/DSEgrp18/baseline-pathnirvana.git repo
python repo/evaluate.py --stage setup
python repo/evaluate.py --stage all --tag v36k

# Run 2 — pin dataset to Version 2 (72k), then:
rm -rf repo /kaggle/working/vits_si
git clone https://github.com/DSEgrp18/baseline-pathnirvana.git repo
python repo/evaluate.py --stage setup
python repo/evaluate.py --stage all --tag v72k
```

Outputs: `/kaggle/working/eval/v36k/` and `/kaggle/working/eval/v72k/`, each with
`results.json`, `results.csv`, `samples_manifest.json`, and per-sentence wavs.

---

## 6. Results

### 6.1 Overall (mean across 16 sentences)

| Model | Mean WER ↓ | Mean CER ↓ | Mean UTMOS ↑ | Mean RTF ↓ |
|-------|-----------|-----------|-------------|-----------|
| VITS @ 36k | 1.010 | 0.833 | 3.180 | 0.064 |
| VITS @ 72k | 1.031 | 0.882 | **3.416** | 0.060 |
| **Δ (72k − 36k)** | +0.021 | +0.049 | **+0.236** | −0.004 |

**Headline:** doubling training (36k → 72k) raised predicted naturalness (**UTMOS +0.24**)
while intelligibility metrics (WER/CER) stayed flat within noise. RTF is ~0.06 for both —
comfortably faster than real-time.

### 6.2 Per-category breakdown

Per-category means (CER lower = better, UTMOS higher = better). **Bold** = better of the two.

| Category | 36k CER | 72k CER | 36k UTMOS | 72k UTMOS |
|----------|--------|--------|-----------|-----------|
| statement | **0.731** | 0.932 | 3.266 | **3.594** |
| question | 0.824 | **0.687** | 2.886 | **3.225** |
| exclaim | **0.882** | 1.146 | 2.716 | **3.324** |
| numbers | **0.798** | 0.901 | **3.555** | 3.474 |
| long | 0.873 | **0.816** | 3.075 | **3.163** |
| codeswitch | 0.772 | **0.734** | **3.413** | 3.408 |
| short | **1.012** | 1.029 | 3.452 | **3.733** |

> codeswitch UTMOS (36k 3.413 vs 72k 3.408) is effectively tied.
>
> **UTMOS rises for 72k in 5 of 7 categories** (`numbers` and `codeswitch` are flat/slightly
> down), the clearest
> and most consistent signal in the whole evaluation. CER is mixed/noisy: 72k wins on
> question / long / codeswitch, 36k wins on statement / exclaim / numbers — no directional
> trend, consistent with single-run stochastic-synthesis noise rather than a real gap.

---

## 7. Interpretation

**1. More training improved naturalness, not intelligibility.**
From 36k → 72k, UTMOS rose **+0.24 overall** and in 5 of 7 categories, while WER and CER
stayed flat (both changed by < 0.05, within single-run noise). This is a textbook VITS
pattern: additional steps refine timbre and smoothness (what UTMOS captures) faster than
they improve phonetic precision. **Net verdict: the 72k model is the better one to ship, and
the model was still improving at 72k** — so continuing toward ~100k steps is justified.

**2. Intelligibility has likely hit a soft ceiling.**
WER is pinned at ~1.0 for both models (saturated — see §8) and CER shows no direction. More
VITS steps alone are unlikely to move intelligibility much; that bottleneck is **data
quantity/quality and grapheme coverage**, not training length. This is the quantitative
evidence for the next tracks — MMS-TTS transfer learning and F5-TTS — described in
[architecture-evolution-and-research-gap.md](architecture-evolution-and-research-gap.md).

**3. Weakest areas.**
By UTMOS, the hardest categories for *both* models are **`long`** (~3.1, long-form prosody
drifts) and **`question`** at 36k (2.89, intonation) — the latter improves most with training
(→ 3.23 at 72k). **`codeswitch`** is a structural failure for both: the VITS vocabulary has no
Latin letters, so English words are silently dropped (§8.2). Best categories are **`short`**
and **`statement`** (short, in-domain utterances).

**4. Caveats on these numbers.**
Each row is a **single stochastic synthesis run** (VITS samples its duration predictor), so
per-sentence and small aggregate differences carry real variance. The UTMOS gain is
consistent enough across categories to trust as directional, but for the final report we
should **average ≥3 seeds per sentence** and, above all, run the **human MOS** (§9) — that is
the number that decides 36k vs 72k perceptually.

---

## 8. Known limitations (report these honestly)

1. **WER is saturated (~1.0) and non-discriminative.** Whisper's Sinhala is weak *and* the
   model is undertrained, so nearly every word round-trips as an error. Do **not** report
   absolute WER as a quality figure; use CER for relative ASR signal and lean on human MOS.
2. **Code-switching is broken by design.** The VITS vocabulary has **no Latin letters**, so
   English words (`laptop`, `email`, `meeting`, `online`) are dropped by the tokenizer
   (`Character 'l' not found in the vocabulary. Discarding it.`). This is a genuine finding
   that motivates the F5-TTS / code-switch track — not a harness bug.
3. **UTMOS is cross-lingual.** It was trained mostly on English speech; treat it as a rough
   naturalness proxy, not ground truth for Sinhala.
4. **Objective ≠ perceptual.** The definitive quality measure for the paper is the
   native-speaker MOS from §9.

---

## 9. Human MOS protocol (planned)

The synthesized wavs under `eval/<model>/` feed a native-speaker listening test:

- **Raters:** N native Sinhala speakers (target N ≥ 10).
- **Scale:** 5-point MOS (1 = bad … 5 = excellent) for **naturalness**; optionally a second
  scale for **intelligibility**.
- **Design:** samples presented in randomized order, model identity hidden (blind),
  36k and 72k of the same sentence not shown adjacently.
- **Report:** mean MOS ± 95% CI per model; a paired test (e.g. Wilcoxon) for 36k vs 72k.
- The `samples_manifest.json` produced by the harness lists every `(model, category, text,
  wav)` and can drive an HTML rating kit.

---

## 10. Artifacts

| File | Contents |
|------|----------|
| `eval/<tag>/results.csv` | Per-model mean WER/CER/UTMOS/RTF (the comparison table) |
| `eval/<tag>/results.json` | Full per-sentence detail (for the per-category table) |
| `eval/<tag>/samples_manifest.json` | Sample list for the listening test |
| `eval/<tag>/<model>/*.wav` | Synthesized audio for human MOS |
