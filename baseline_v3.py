#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinhala F5-TTS fine-tuning pipeline for Kaggle (PathNirvana dataset) — Track B.

This is the F5-TTS counterpart to baseline_v2.py (VITS). Instead of training a
model from scratch, it *fine-tunes* the pretrained F5-TTS foundation model on
Sinhala, so the output inherits the base model's naturalness/prosody and (via a
reference clip at inference) its expressiveness. Same discipline as v2:
staged CLI, --smoke verification, cross-session resume.

============================ HONEST STATUS ============================
This wraps F5-TTS's OFFICIAL tooling (prepare_csv_wavs + f5-tts_finetune-cli +
f5-tts_infer-cli). It is grounded in the real CLI (args verified from source),
but F5-TTS is version-sensitive and this has NOT been run end-to-end yet, so
expect 1-3 shakeout iterations on Kaggle (exactly like v2 needed). The two
riskiest parts are marked `# RISK:` below:
  1. Sinhala vocab extension (base model has never seen Sinhala characters).
  2. Redirecting F5-TTS's package-relative ckpts/ and data/ dirs into
     /kaggle/working via symlinks so they persist and resume.
Full fine-tuning (lr=1e-5), NOT LoRA — LoRA is not in mainline F5-TTS. The
proposal's LoRA/PEFT plan is a later custom addition; revisit once this runs.
======================================================================

USAGE on Kaggle (GPU T4, Internet ON), one cell each:
    !rm -rf repo && git clone https://github.com/DSEgrp18/baseline-pathnirvana.git repo
    !python repo/baseline_v3.py --stage setup
    !python repo/baseline_v3.py --stage all --smoke     # verify first!
    !python repo/baseline_v3.py --stage all             # real fine-tuning

Resume across sessions: attach the previous version output as an Input; the
symlinked ckpts dir exposes model_last.pt and F5-TTS resumes from it.
NOTE: F5-TTS fine-tuning is heavier than VITS — budget your 30 h/week quota.
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# ----------------------------------------------------------------------------
REPO = 'pnfo/sinhala-tts-dataset'
TAG = 'v2.1'
SEED = 1234
SR = 24000                     # F5-TTS operates at 24 kHz (VITS/v2 used 22.05 kHz)
EXP_NAME = 'F5TTS_v1_Base'     # base architecture/checkpoint family
DATASET_NAME = 'si_f5'         # F5-TTS keys ckpts/data dirs by this name
TOKENIZER = 'custom'           # custom vocab = base vocab + Sinhala chars
HF_REPO = 'SWivid/F5-TTS'
BASE_CKPT_REL = 'F5TTS_v1_Base/model_1250000.safetensors'
BASE_VOCAB_REL = 'F5TTS_v1_Base/vocab.txt'

ON_KAGGLE = os.path.exists('/kaggle')
_HERE = os.path.dirname(os.path.abspath(__file__))
TMP = '/kaggle/tmp' if ON_KAGGLE else os.path.join(_HERE, '.tmp')
WORK = '/kaggle/working' if ON_KAGGLE else os.path.join(_HERE, '.work')

RAW = f'{TMP}/pathnirvana_raw'          # downloaded PathNirvana (ephemeral)
CACHE = f'{TMP}/f5_cache'               # base ckpt + vocab download cache
DATA = f'{TMP}/f5_si'                   # 24 kHz wavs + metadata.csv
PREP = f'{TMP}/f5_si_prepared'          # raw.arrow + duration.json + vocab.txt
VOCAB = f'{TMP}/f5_si_vocab.txt'        # extended vocab we pass to the trainer
CKPTS = f'{WORK}/f5_ckpts'              # persisted checkpoints (symlink target)
SAMPLES = f'{WORK}/samples_f5'
EXPORT = f'{WORK}/export_f5'

TEST_SENTENCES = [
    'අද දවස ඉතාම සුන්දරයි.',
    'ඔයාට හෙට උදේ මෙතනට එන්න පුළුවන්ද?',
    'අනේ! මේක නම් හරිම පුදුමයි!',
    'ඉස්සර කාලයේ, එක ගමක, හොඳ සිතක් ඇති, දුප්පත් ගොවියෙක් ජීවත් වුණා.',
]


def sh(cmd):
    print('+', cmd)
    subprocess.run(cmd, shell=True, check=True)


# ----------------------------------------------------------------------------
# stage: setup
# ----------------------------------------------------------------------------
def _base_files():
    """Download (idempotently) the base checkpoint + vocab, return their paths."""
    from huggingface_hub import hf_hub_download
    os.makedirs(CACHE, exist_ok=True)
    ckpt = hf_hub_download(HF_REPO, BASE_CKPT_REL, local_dir=CACHE)
    vocab = hf_hub_download(HF_REPO, BASE_VOCAB_REL, local_dir=CACHE)
    return ckpt, vocab


def stage_setup(cfg):
    py = sys.executable
    sh(f'{py} -m pip -q install f5-tts librosa soundfile pandas pyloudnorm huggingface_hub')
    # pre-fetch base weights so the train stage doesn't stall on a huge download
    ckpt, vocab = _base_files()
    print('base checkpoint:', ckpt)
    print('base vocab:', vocab)
    sh(f'{py} -c "import f5_tts; print(\'f5-tts OK\')"')
    print('Setup complete.')


# ----------------------------------------------------------------------------
# stage: data  (reuse PathNirvana, resample to 24 kHz, build F5 dataset)
# ----------------------------------------------------------------------------
def _download_dataset():
    import requests
    os.makedirs(RAW, exist_ok=True)
    meta = f'{RAW}/metadata.csv'
    if not os.path.exists(meta):
        r = requests.get(f'https://raw.githubusercontent.com/{REPO}/{TAG}/metadata.csv', timeout=60)
        r.raise_for_status()
        open(meta, 'wb').write(r.content)
    if glob.glob(f'{RAW}/**/sinh_*.wav', recursive=True):
        return
    headers = {'Accept': 'application/vnd.github+json'}
    if os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    assets = []
    for _ in range(3):
        rel = requests.get(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}',
                           headers=headers, timeout=60).json()
        assets = rel.get('assets', [])
        if assets:
            break
        import time
        print('GitHub API empty (rate limit?), retrying in 30 s...')
        time.sleep(30)
    assert assets, 'Could not list release assets. Set GITHUB_TOKEN and retry.'
    archive = max(assets, key=lambda a: a['size'])
    print(f"Downloading {archive['name']} ({archive['size'] // 2**20} MB)...")
    arc = f'{TMP}/wavs_archive'
    with requests.get(archive['browser_download_url'], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(arc, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    if zipfile.is_zipfile(arc):
        with zipfile.ZipFile(arc) as z:
            z.extractall(RAW)
    else:
        with tarfile.open(arc) as t:
            t.extractall(RAW)
    os.remove(arc)


def _locate_dataset(cfg):
    dirs = ([cfg.dataset_dir] if cfg.dataset_dir else []) + \
           (['/kaggle/input'] if os.path.exists('/kaggle/input') else [])
    for d in dirs:
        metas = glob.glob(f'{d}/**/metadata.csv', recursive=True)
        wavs = glob.glob(f'{d}/**/sinh_*.wav', recursive=True)
        if metas and wavs:
            print(f'Using attached dataset under {d} (no download).')
            return metas[0], os.path.dirname(sorted(wavs)[0])
    _download_dataset()
    metas = glob.glob(f'{RAW}/**/metadata.csv', recursive=True) or [f'{RAW}/metadata.csv']
    wavs = glob.glob(f'{RAW}/**/sinh_*.wav', recursive=True)
    assert wavs, 'no sinh_*.wav found after extraction'
    return metas[0], os.path.dirname(sorted(wavs)[0])


def _clean_text(t):
    t = ' '.join(t.strip().replace('-පෙ-', '').split())
    if t and t[-1] not in '.?!':
        t += '.'
    return t


def process_clip(args):
    """Load -> 24 kHz mono -> trim -> loudness-normalize -> pad -> 16-bit PCM."""
    src, dst = args
    import librosa
    import numpy as np
    import soundfile as sf
    try:
        y, _ = librosa.load(src, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=40)
        if len(y) < int(0.35 * SR):
            return None
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(SR)
            y = pyln.normalize.loudness(y, meter.integrated_loudness(y), -24.0)
        except Exception:
            pass
        peak = float(np.abs(y).max())
        if peak > 0.99:
            y *= 0.99 / peak
        y = np.concatenate([np.zeros(int(0.05 * SR)), y, np.zeros(int(0.10 * SR))]).astype('float32')
        dur = len(y) / SR
        if not 1.0 <= dur <= 12.0:
            return None
        sf.write(dst, y, SR, subtype='PCM_16')
        return round(dur, 3)
    except Exception:
        return None


def _build_extended_vocab(texts, base_vocab_path):
    """RISK: F5-TTS base vocab has no Sinhala. Append Sinhala characters that
    are missing so their embeddings can be learned (base rows are preserved)."""
    base = [ln.rstrip('\n') for ln in open(base_vocab_path, encoding='utf-8')]
    present = set(base)
    extra = [c for c in sorted(set(''.join(texts))) if c not in present]
    with open(VOCAB, 'w', encoding='utf-8') as f:
        f.write('\n'.join(base + extra) + '\n')
    print(f'vocab: {len(base)} base + {len(extra)} new Sinhala chars = {len(base) + len(extra)}')
    return VOCAB


def stage_data(cfg):
    from concurrent.futures import ProcessPoolExecutor

    if os.path.exists(f'{PREP}/raw.arrow'):
        print('Prepared F5 dataset already present, skipping data stage.')
        return

    _, base_vocab = _base_files()
    meta_path, wav_dir = _locate_dataset(cfg)

    rows = []
    with open(meta_path, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('|')
            if len(p) == 4 and p[3] == cfg.speaker:
                text = _clean_text(p[2])
                if text:
                    rows.append((p[0], text))
    rows = [(i, t) for i, t in rows if os.path.exists(f'{wav_dir}/{i}.wav')]
    random.seed(SEED)
    random.shuffle(rows)
    if cfg.smoke:
        rows = rows[:cfg.smoke_clips]
    print(f'{len(rows)} clips for speaker "{cfg.speaker}"')

    os.makedirs(f'{DATA}/wavs', exist_ok=True)
    jobs = [(f'{wav_dir}/{i}.wav', f'{DATA}/wavs/{i}.wav') for i, _ in rows]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as ex:
        durs = list(ex.map(process_clip, jobs, chunksize=16))
    kept = [(i, t) for (i, t), d in zip(rows, durs) if d is not None]
    hours = sum(d for d in durs if d) / 3600
    print(f'kept {len(kept)}/{len(rows)} clips | {hours:.2f} h @ 24 kHz')

    # F5-TTS metadata.csv:  <absolute wav path>|<text>
    with open(f'{DATA}/metadata.csv', 'w', encoding='utf-8') as f:
        for i, t in kept:
            f.write(f'{DATA}/wavs/{i}.wav|{t}\n')

    _build_extended_vocab([t for _, t in kept], base_vocab)

    # Official prep: build raw.arrow + duration.json from the wavs+metadata.
    # RISK: prepare_csv_wavs signature is version-sensitive (input dir vs csv);
    # this uses the current main form: <input_dir> <output_dir>.
    os.makedirs(PREP, exist_ok=True)
    sh(f'{sys.executable} -m f5_tts.train.datasets.prepare_csv_wavs "{DATA}" "{PREP}"')
    # prefer our extended vocab over the one prepare generated from data alone
    shutil.copy(VOCAB, f'{PREP}/vocab.txt')

    os.makedirs(WORK, exist_ok=True)
    json.dump({'speaker': cfg.speaker, 'clips': len(kept), 'hours': round(hours, 2),
               'sr': SR, 'smoke': cfg.smoke}, open(f'{WORK}/data_stats_f5.json', 'w'))


# ----------------------------------------------------------------------------
# stage: train
# ----------------------------------------------------------------------------
def _pkg_relative(sub):
    """Resolve F5-TTS's package-relative <root>/<sub> dir (ckpts or data),
    which the CLI hardcodes as files('f5_tts')/../../<sub>."""
    import f5_tts
    return Path(f5_tts.__file__).resolve().parent.parent.parent / sub


def _link_into_working():
    """RISK: F5-TTS writes ckpts/ and data/ next to its package (ephemeral on
    Kaggle). Symlink both to persistent locations so training saves survive and
    resume works, and so the trainer finds our prepared dataset."""
    # data/<dataset>_<tokenizer>  ->  our PREP dir
    data_link = _pkg_relative('data') / f'{DATASET_NAME}_{TOKENIZER}'
    ckpt_link = _pkg_relative('ckpts') / DATASET_NAME
    os.makedirs(CKPTS, exist_ok=True)
    real_ckpt = f'{CKPTS}/{DATASET_NAME}'
    os.makedirs(real_ckpt, exist_ok=True)
    for link, target in [(data_link, PREP), (ckpt_link, real_ckpt)]:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                shutil.rmtree(link)
        os.symlink(target, link)
        print(f'linked {link} -> {target}')
    return real_ckpt


def stage_train(cfg):
    base_ckpt, _ = _base_files()
    real_ckpt = _link_into_working()

    # Passing --pretrain each run is safe: F5-TTS resumes from model_last.pt if
    # it exists in the ckpts dir, otherwise starts from the pretrain weights.
    resume = os.path.exists(f'{real_ckpt}/model_last.pt')
    print('RESUMING from model_last.pt' if resume else 'Fresh fine-tune from base.')

    if cfg.smoke:
        knobs = ('--epochs 1 --num_warmup_updates 2 --save_per_updates 5 '
                 '--last_per_updates 5 --keep_last_n_checkpoints 1 '
                 '--batch_size_per_gpu 1600')
    else:
        knobs = (f'--epochs {cfg.epochs} --num_warmup_updates 2000 '
                 '--save_per_updates 5000 --last_per_updates 1000 '
                 '--keep_last_n_checkpoints 2 --batch_size_per_gpu 3200')

    cmd = (
        'f5-tts_finetune-cli '
        f'--exp_name {EXP_NAME} '
        f'--dataset_name {DATASET_NAME} '
        '--finetune '
        f'--pretrain "{base_ckpt}" '
        f'--tokenizer {TOKENIZER} --tokenizer_path "{VOCAB}" '
        f'--learning_rate {cfg.lr} '
        '--batch_size_type frame '
        f'--grad_accumulation_steps {cfg.grad_accum} '
        f'{knobs}'
    )
    sh(cmd)


# ----------------------------------------------------------------------------
# stage: infer / package
# ----------------------------------------------------------------------------
def _latest_ckpt():
    real_ckpt = f'{CKPTS}/{DATASET_NAME}'
    for name in ('model_last.pt',):
        if os.path.exists(f'{real_ckpt}/{name}'):
            return f'{real_ckpt}/{name}'
    cks = sorted(glob.glob(f'{real_ckpt}/model_*.pt'))
    assert cks, f'no checkpoint under {real_ckpt}'
    return cks[-1]


def _reference_clip():
    """F5-TTS needs a reference audio + its transcript to condition on. Use the
    first processed training clip and its text."""
    wavs = sorted(glob.glob(f'{DATA}/wavs/*.wav'))
    assert wavs, 'no processed wavs for a reference clip'
    ref_wav = wavs[0]
    ref_id = os.path.splitext(os.path.basename(ref_wav))[0]
    ref_text = ''
    with open(f'{DATA}/metadata.csv', encoding='utf-8') as f:
        for line in f:
            path, text = line.rstrip('\n').split('|', 1)
            if ref_id in path:
                ref_text = text
                break
    return ref_wav, ref_text


def stage_infer(cfg):
    ckpt = _latest_ckpt()
    ref_wav, ref_text = _reference_clip()
    os.makedirs(SAMPLES, exist_ok=True)
    print('Synthesizing with', ckpt)
    for k, sent in enumerate(TEST_SENTENCES):
        out_dir = f'{SAMPLES}/s{k}'
        cmd = (
            'f5-tts_infer-cli '
            f'--model {EXP_NAME} '
            f'--ckpt_file "{ckpt}" '
            f'--vocab_file "{VOCAB}" '
            f'--ref_audio "{ref_wav}" '
            f'--ref_text "{ref_text}" '
            f'--gen_text "{sent}" '
            f'--output_dir "{out_dir}"'
        )
        try:
            sh(cmd)
            print(f'  sample {k} <- {sent}')
        except subprocess.CalledProcessError:
            print(f'  ! inference failed for sample {k} (see log above)')


def stage_package(cfg):
    os.makedirs(EXPORT, exist_ok=True)
    for src in [_latest_ckpt(), VOCAB, f'{WORK}/data_stats_f5.json']:
        if os.path.exists(src):
            shutil.copy(src, EXPORT)
    print('Export contents:', os.listdir(EXPORT))
    print('\nNow: Save Version (Quick Save or Save & Run All). Next session, attach '
          'this version output as an Input to resume from model_last.pt.')


# ----------------------------------------------------------------------------
STAGES = {'setup': stage_setup, 'data': stage_data, 'train': stage_train,
          'infer': stage_infer, 'package': stage_package}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', default='all', choices=list(STAGES) + ['all'])
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--speaker', default='mettananda', choices=['mettananda', 'oshadi'])
    p.add_argument('--dataset-dir', default=None)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--grad-accum', type=int, default=1)
    cfg = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')   # Kaggle T4 x2 -> pin one
    cfg.smoke_clips = 60

    os.makedirs(TMP, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    if cfg.stage == 'all':
        for name in ('data', 'train', 'infer', 'package'):
            print(f'\n========== stage: {name} ==========')
            STAGES[name](cfg)
    else:
        STAGES[cfg.stage](cfg)


if __name__ == '__main__':
    main()
