#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Sinhala TTS checkpoints — objective metrics + samples for human MOS.

For each checkpoint it synthesizes a FIXED held-out test set, then reports:
  - WER / CER   : Whisper transcribes the synthesized audio; compared to the
                  input text. Lower = more intelligible. (Sinhala Whisper is
                  weak, so treat ABSOLUTE numbers with caution but RELATIVE
                  comparisons — 36k vs 72k — as valid.)
  - UTMOS       : a neural predicted-naturalness score (~1-5). Rough relative
                  signal only; trained mostly on English. NOT a substitute for
                  human MOS in the paper.
  - RTF         : real-time factor (synth time / audio duration). <1 = faster
                  than real-time.
It also saves every synthesized wav under eval/<label>/ + a samples manifest,
ready for a native-speaker listening test (human MOS).

USAGE on Kaggle (GPU T4, Internet ON), one cell each:
    !python repo/evaluate.py --stage setup
    !python repo/evaluate.py --stage all            # evaluates every checkpoint found
    !python repo/evaluate.py --stage all --smoke    # 4 sentences, quick sanity check

Checkpoints are auto-discovered: attach the dataset(s) with your run folders
(e.g. vits-si-ckpt-36k and vits-si-ckpt-72k) and it evaluates the latest
checkpoint in EACH, labeled by its folder — so 36k vs 72k compare side by side.
Or pass explicit paths: --checkpoints "/kaggle/input/a/....pth,/kaggle/input/b/....pth"
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import unicodedata

ON_KAGGLE = os.path.exists('/kaggle')
_HERE = os.path.dirname(os.path.abspath(__file__))
WORK = '/kaggle/working' if ON_KAGGLE else os.path.join(_HERE, '.work')
OUT = f'{WORK}/eval'

# Punctuation stripped before computing WER/CER (compare words, not marks).
_PUNCT = set("!'()\",-.:;?‘’“”।…" + '‍')  # incl. ZWJ

# Fixed held-out test set — keep IDENTICAL across all models (this is the
# benchmark). Categories let you report per-type breakdowns.
TEST_SET = [
    ('statement', 'අද දවස ඉතාම සුන්දරයි.'),
    ('statement', 'මම හෙට උදේ පාසල් යනවා.'),
    ('statement', 'ශ්‍රී ලංකාව ලස්සන දිවයිනකි.'),
    ('question',  'ඔයාට හෙට උදේ මෙතනට එන්න පුළුවන්ද?'),
    ('question',  'මේ පොත කාගේද?'),
    ('question',  'ඔබ කොහෙද යන්නේ?'),
    ('exclaim',   'අනේ! මේක නම් හරිම පුදුමයි!'),
    ('exclaim',   'අපොයි! මට බය හිතුණා!'),
    ('numbers',   'මේ පොතේ පිටු දෙසිය පනහක් තියෙනවා.'),
    ('numbers',   'අද දිනය දෙදහස් විසිහයයි.'),
    ('long',      'ඉස්සර කාලයේ, එක ගමක, හොඳ සිතක් ඇති, දුප්පත් ගොවියෙක් ජීවත් වුණා.'),
    ('long',      'ඔහු සෑම දිනකම උදෑසන අවදි වී, තම කුඹුරට ගොස්, දහවල් වන තුරු වෙහෙස මහන්සි වී වැඩ කළේය.'),
    ('codeswitch','මම laptop එකෙන් email එකක් යැව්වා.'),
    ('codeswitch','අද meeting එක online තියෙනවා.'),
    ('short',     'ස්තූතියි.'),
    ('short',     'සුබ උදෑසනක්.'),
]


def sh(cmd):
    print('+', cmd)
    subprocess.run(cmd, shell=True, check=True)


def stage_setup(cfg):
    py = sys.executable
    sh(f'{py} -m pip -q install coqui-tts jiwer soundfile librosa "transformers==4.53.0"')
    sh(f'{py} -m pip -q install -U openai-whisper')
    # coqui-tts downgrades torch to 2.8.0, but Kaggle keeps its torchvision/
    # torchaudio built for a newer torch -> "operator torchvision::nms does not
    # exist" crashes the transformers/TTS import chain. Re-pin the vision+audio
    # libs to the pair that matches torch 2.8.0 so everything imports cleanly.
    sh(f'{py} -m pip -q install torchvision==0.23.0 torchaudio==2.8.0')
    print('Setup complete. (UTMOS loads on demand via torch.hub.)')


# ---------------------------------------------------------------------------
def _discover(cfg):
    """Return [(label, ckpt_path, config_path)]. Explicit --checkpoints wins;
    else the latest checkpoint of each distinct run folder that has a config."""
    out = []
    if cfg.checkpoints:
        for p in [c.strip() for c in cfg.checkpoints.split(',') if c.strip()]:
            cfgp = os.path.join(os.path.dirname(p), 'config.json')
            label = os.path.basename(os.path.dirname(p))[:24] + '/' + os.path.basename(p)
            out.append((label, p, cfgp))
        return out

    run_dirs = sorted({os.path.dirname(p) for p in
                       glob.glob('/kaggle/input/**/config.json', recursive=True) +
                       glob.glob(f'{WORK}/**/config.json', recursive=True)})
    for d in run_dirs:
        ckpts = glob.glob(f'{d}/checkpoint_*.pth')
        if ckpts:
            latest = max(ckpts, key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))
        elif os.path.exists(f'{d}/best_model.pth'):
            latest = f'{d}/best_model.pth'
        else:
            continue
        # label by the parent dataset/run folder so 36k vs 72k are distinguishable
        label = os.path.basename(os.path.dirname(d)) or os.path.basename(d)
        out.append((label, latest, f'{d}/config.json'))
    return out


def _norm(t):
    t = unicodedata.normalize('NFC', t)
    t = ''.join(ch for ch in t if ch not in _PUNCT)
    return ' '.join(t.split())


def _load_asr(cfg):
    try:
        import whisper
        print(f'loading Whisper ({cfg.asr_model}) ...')
        return whisper.load_model(cfg.asr_model)
    except Exception as e:
        print('! Whisper unavailable — WER/CER will be skipped:', e)
        return None


def _load_utmos():
    try:
        import torch
        print('loading UTMOS predictor ...')
        return torch.hub.load('tarepan/SpeechMOS', 'utmos22_strong', trust_repo=True)
    except Exception as e:
        print('! UTMOS unavailable — predicted MOS will be skipped:', e)
        return None


def stage_eval(cfg):
    import torch
    import jiwer
    import torchaudio
    from TTS.utils.synthesizer import Synthesizer

    targets = _discover(cfg)
    assert targets, 'no checkpoints found — attach a run dataset or pass --checkpoints'
    print('evaluating:')
    for label, ck, _ in targets:
        print(f'  {label:28s} <- {ck}')

    sentences = TEST_SET[:4] if cfg.smoke else TEST_SET
    asr = _load_asr(cfg)
    utmos = _load_utmos()
    use_cuda = torch.cuda.is_available()
    os.makedirs(OUT, exist_ok=True)

    results = {}
    for label, ckpt, confp in targets:
        print(f'\n===== {label} =====')
        try:
            syn = Synthesizer(tts_checkpoint=ckpt, tts_config_path=confp, use_cuda=use_cuda)
        except Exception as e:
            print('! failed to load, skipping:', e)
            continue
        sr = syn.output_sample_rate
        sub = f'{OUT}/{label}'.replace(' ', '_')
        os.makedirs(sub, exist_ok=True)

        rows = []
        for i, (cat, text) in enumerate(sentences):
            wav_path = f'{sub}/{i:02d}_{cat}.wav'
            t0 = time.time()
            wav = syn.tts(text)
            synth_t = time.time() - t0
            syn.save_wav(wav, wav_path)
            dur = len(wav) / sr
            rtf = synth_t / dur if dur else float('nan')

            wer = cer = mos = None
            if asr is not None:
                try:
                    hyp = asr.transcribe(wav_path, language='si')['text']
                    wer = jiwer.wer(_norm(text), _norm(hyp))
                    cer = jiwer.cer(_norm(text), _norm(hyp))
                except Exception as e:
                    print('  asr error:', e)
            if utmos is not None:
                try:
                    w, wsr = torchaudio.load(wav_path)
                    mos = float(utmos(w, wsr).item())
                except Exception as e:
                    print('  utmos error:', e)

            rows.append(dict(idx=i, category=cat, text=text, wav=wav_path,
                             wer=wer, cer=cer, utmos=mos, rtf=rtf))
            print(f'  [{cat:10s}] wer={_fmt(wer)} cer={_fmt(cer)} '
                  f'utmos={_fmt(mos)} rtf={rtf:.2f}')

        agg = _aggregate(rows)
        results[label] = dict(checkpoint=ckpt, n=len(rows), **agg, samples=rows)
        print(f'  --> mean WER={_fmt(agg["wer"])} CER={_fmt(agg["cer"])} '
              f'UTMOS={_fmt(agg["utmos"])} RTF={_fmt(agg["rtf"])}')

    json.dump(results, open(f'{OUT}/results.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    _write_report(results)


def _fmt(x):
    return '  -  ' if x is None else f'{x:.3f}'


def _aggregate(rows):
    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else None
    return dict(wer=mean('wer'), cer=mean('cer'), utmos=mean('utmos'), rtf=mean('rtf'))


def _write_report(results):
    # comparison CSV + a plaintext table
    lines = ['model,mean_wer,mean_cer,mean_utmos,mean_rtf,n']
    for label, r in results.items():
        lines.append(f'{label},{_c(r["wer"])},{_c(r["cer"])},'
                     f'{_c(r["utmos"])},{_c(r["rtf"])},{r["n"]}')
    open(f'{OUT}/results.csv', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    # samples manifest for the human listening test
    manifest = []
    for label, r in results.items():
        for s in r['samples']:
            manifest.append(dict(model=label, category=s['category'],
                                 text=s['text'], wav=s['wav']))
    json.dump(manifest, open(f'{OUT}/samples_manifest.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    print('\n================ COMPARISON ================')
    print(f'{"model":26s} {"WER":>7} {"CER":>7} {"UTMOS":>7} {"RTF":>7}')
    for label, r in results.items():
        print(f'{label[:26]:26s} {_fmt(r["wer"]):>7} {_fmt(r["cer"]):>7} '
              f'{_fmt(r["utmos"]):>7} {_fmt(r["rtf"]):>7}')
    print('============================================')
    print(f'\nWrote: {OUT}/results.json, results.csv, samples_manifest.json')
    print(f'Synthesized wavs under {OUT}/<model>/ — use these for the human MOS test.')
    print('\nReminder: report WER (relative), UTMOS (rough), AND human MOS together.')


def _c(x):
    return '' if x is None else f'{x:.4f}'


STAGES = {'setup': stage_setup, 'eval': stage_eval}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', default='all', choices=list(STAGES) + ['all'])
    p.add_argument('--smoke', action='store_true', help='4 sentences only')
    p.add_argument('--checkpoints', default=None,
                   help='comma-separated .pth paths; omit to auto-discover attached runs')
    p.add_argument('--asr-model', default='large-v3',
                   help='Whisper model for intelligibility (e.g. large-v3, medium)')
    cfg = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # Kaggle T4 x2 -> one GPU
    os.makedirs(WORK, exist_ok=True)

    if cfg.stage == 'all':
        for name in ('eval',):
            STAGES[name](cfg)
    else:
        STAGES[cfg.stage](cfg)


if __name__ == '__main__':
    main()
