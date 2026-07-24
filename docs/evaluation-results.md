# Sinhala TTS — Evaluation Pipeline & Results

**Project:** Development of a Natural-Sounding Sinhala TTS System (UoM CS3501, Group 18 / P15)
**Track evaluated:** Track A — VITS from scratch (Coqui-TTS) on the PathNirvana dataset
**Harness:** [`evaluate.py`](../evaluate.py)
**Status:** _results pending — populate the tables below from the two `results.csv` runs_

---

## 1. Purpose

Quantify how our from-scratch Sinhala VITS model improves with additional training, and
produce audio samples for a native-speaker listening test. We compare two training
checkpoints of the **same** model:

| Label | Checkpoint | Effective training |
|-------|-----------|--------------------|
| `v36k` | `checkpoint_36000.pth` (dataset Version 1) | ~36k steps |
| `v72k` | latest checkpoint (dataset Version 2) | ~72k effective steps (warm-restart continued) |

> **Note on "effective steps":** the step counter resets to 0 on a warm *restore*, so the
> 72k model's file may still read `checkpoint_36000.pth`. Effective steps = commits × ~36k.

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
| Harness commit | _fill in `git rev-parse --short HEAD`_ |

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

_Paste the two `results.csv` rows here._

| Model | Mean WER ↓ | Mean CER ↓ | Mean UTMOS ↑ | Mean RTF ↓ |
|-------|-----------|-----------|-------------|-----------|
| VITS @ 36k (`v36k`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| VITS @ 72k (`v72k`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Δ (72k − 36k)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### 6.2 Per-category breakdown

_Optional — extract per-category means from `results.json` if you want this table._

| Category | 36k CER | 72k CER | 36k UTMOS | 72k UTMOS |
|----------|--------|--------|-----------|-----------|
| statement | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| question | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| exclaim | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| numbers | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| long | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| codeswitch | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| short | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 7. Interpretation

_Written after the numbers land. Key questions to answer:_

- **Did more training help?** Compare CER and UTMOS (36k → 72k). A meaningful improvement
  means the model is still learning → continue toward ~100k steps.
- **Or have we hit the ceiling?** If 36k ≈ 72k, extra VITS steps won't help — the bottleneck
  is data quantity/quality or architecture, which is the evidence for pivoting to
  MMS-TTS transfer learning or the F5-TTS track (see
  [architecture-evolution-and-research-gap.md](architecture-evolution-and-research-gap.md)).
- **Where is quality weakest?** Use the per-category table to find the worst category
  (typically `long` for coherence and `codeswitch` for the vocab gap).

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
