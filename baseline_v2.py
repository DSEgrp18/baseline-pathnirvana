#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinhala VITS training pipeline for Kaggle (PathNirvana dataset).

Improvements over baseline.py:
  - Runs as a script (no Colab magics, no google.colab) -> works on Kaggle.
  - Smoke-test mode (--smoke): full end-to-end run on ~100 clips / 3 epochs
    in ~15-25 min so you verify EVERYTHING works before spending GPU quota.
  - Crash-safe resume: auto-detects checkpoints in /kaggle/working (same
    session) or /kaggle/input (previous notebook version output) and continues.
  - Audio hygiene: 22.05k mono, silence trim with padded onsets/endings
    (clean stops), loudness normalization to -24 LUFS, duration filter 1-12s.
  - Text hygiene: punctuation preserved in vocab, guaranteed sentence-final
    punctuation (prevents trailing babble / bad endings).
  - Optional --phonemes: espeak-ng Sinhala phoneme input (proposal experiment E2).

USAGE in a Kaggle notebook (GPU T4, Internet ON), one cell each:
    !git clone https://github.com/<you>/baseline-pathnirvana.git repo
    !python repo/kaggle_pipeline.py --stage setup
    !python repo/kaggle_pipeline.py --stage all --smoke     # verify first!
    !python repo/kaggle_pipeline.py --stage all             # real training

SESSION WORKFLOW (Kaggle wipes /kaggle/working between interactive sessions):
  1. Train interactively for the session (leave ~20 min margin before the
     12 h limit; checkpoints save every save_step automatically).
  2. File > Save Version > "Quick Save" -> preserves /kaggle/working output.
  3. Next session: Add Input -> your own notebook's previous version output.
     This script finds the checkpoint there and resumes automatically.
  IMPORTANT: keep preprocessing flags identical across sessions (same speaker,
  same --phonemes setting) so the vocabulary matches the checkpoint.
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

# ----------------------------------------------------------------------------
# constants / paths
# ----------------------------------------------------------------------------
REPO = 'pnfo/sinhala-tts-dataset'
TAG = 'v2.1'
SEED = 1234
SR = 22050
PUNCT = "!'(),-.:;?‘’“”\" "

ON_KAGGLE = os.path.exists('/kaggle')
_HERE = os.path.dirname(os.path.abspath(__file__))
TMP = '/kaggle/tmp' if ON_KAGGLE else os.path.join(_HERE, '.tmp')
WORK = '/kaggle/working' if ON_KAGGLE else os.path.join(_HERE, '.work')

RAW = f'{TMP}/pathnirvana_raw'      # downloaded archive + extracted wavs (ephemeral)
DATA = f'{TMP}/si_tts'              # processed wavs + manifests (ephemeral, rebuilt per session)
OUT = f'{WORK}/vits_si'             # training runs / checkpoints (persist via Quick Save)
EXPORT = f'{WORK}/export'           # small, clean artifact of the session

TEST_SENTENCES = [
    'අද දවස ඉතාම සුන්දරයි.',                                   # plain statement
    'ඔයාට හෙට උදේ මෙතනට එන්න පුළුවන්ද?',                       # question
    'අනේ! මේක නම් හරිම පුදුමයි!',                               # exclamation
    'ඉස්සර කාලයේ, එක ගමක, හොඳ සිතක් ඇති, දුප්පත් ගොවියෙක් ජීවත් වුණා.',  # commas / pauses
]


def sh(cmd):
    print('+', cmd)
    subprocess.run(cmd, shell=True, check=True)


# ----------------------------------------------------------------------------
# stage: setup  (runs pip in a subprocess; later stages run in fresh python
# processes via `!python`, so no kernel restart is needed on Kaggle)
# ----------------------------------------------------------------------------
def stage_setup(cfg):
    py = sys.executable
    sh(f'{py} -m pip -q uninstall -y torchvision')
    sh(f'{py} -m pip -q install coqui-tts librosa soundfile pandas pyloudnorm "transformers==4.53.0"')
    if cfg.phonemes:
        sh('apt-get -y -qq install espeak-ng')
    # verify in a FRESH process (mimics how later stages will import it)
    sh(f'{py} -c "import TTS, trainer; print(\'coqui-tts OK\')"')
    print('Setup complete. Run the data/train stages as separate !python calls.')


# ----------------------------------------------------------------------------
# stage: data
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
        return  # already extracted this session

    headers = {'Accept': 'application/vnd.github+json'}
    if os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    assets = []
    for attempt in range(3):  # unauthenticated API is rate-limited on shared Kaggle IPs
        rel = requests.get(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}',
                           headers=headers, timeout=60).json()
        assets = rel.get('assets', [])
        if assets:
            break
        print('GitHub API returned no assets (rate limit?), retrying in 30 s...')
        import time
        time.sleep(30)
    assert assets, 'Could not list release assets. Set GITHUB_TOKEN in Kaggle secrets and retry.'

    archive = max(assets, key=lambda a: a['size'])  # wav archive = biggest asset
    print(f"Downloading {archive['name']} ({archive['size'] // 2**20} MB)...")
    arc_path = f'{TMP}/wavs_archive'
    with requests.get(archive['browser_download_url'], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(arc_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    if zipfile.is_zipfile(arc_path):
        with zipfile.ZipFile(arc_path) as z:
            z.extractall(RAW)
    else:
        with tarfile.open(arc_path) as t:
            t.extractall(RAW)
    os.remove(arc_path)


def _locate_dataset(cfg):
    """Return (metadata_path, wav_dir). Prefer a dataset already present on the
    machine -- an explicit --dataset-dir (e.g. a kagglehub download path) or an
    attached Kaggle Input -- so we skip the ~700 MB GitHub download and its
    rate-limit fragility. Fall back to downloading only if nothing is attached."""
    search_dirs = []
    if cfg.dataset_dir:
        search_dirs.append(cfg.dataset_dir)
    if os.path.exists('/kaggle/input'):
        search_dirs.append('/kaggle/input')
    for d in search_dirs:
        metas = glob.glob(f'{d}/**/metadata.csv', recursive=True)
        wavs = glob.glob(f'{d}/**/sinh_*.wav', recursive=True)
        if metas and wavs:
            print(f'Using attached dataset under {d} (no download).')
            return metas[0], os.path.dirname(sorted(wavs)[0])

    _download_dataset()
    metas = glob.glob(f'{RAW}/**/metadata.csv', recursive=True) or [f'{RAW}/metadata.csv']
    wavs = glob.glob(f'{RAW}/**/sinh_*.wav', recursive=True)
    assert wavs, 'no sinh_*.wav files found after extraction'
    return metas[0], os.path.dirname(sorted(wavs)[0])


def _clean_text(t):
    t = t.strip().replace('-පෙ-', '')          # drop repetition marker
    t = ' '.join(t.split())
    if t and t[-1] not in '.?!':               # guaranteed final punctuation -> clean stops
        t += '.'
    return t


def process_clip(args):
    """Load -> trim silence -> loudness-normalize -> pad -> write 16-bit PCM."""
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
            pass                                # fall through to peak guard only
        peak = float(np.abs(y).max())
        if peak > 0.99:
            y = y * (0.99 / peak)
        # 50 ms onset pad, 100 ms ending pad -> model learns clean starts/stops
        y = np.concatenate([np.zeros(int(0.05 * SR)), y, np.zeros(int(0.10 * SR))]).astype('float32')
        dur = len(y) / SR
        if not 1.0 <= dur <= 12.0:
            return None
        sf.write(dst, y, SR, subtype='PCM_16')
        return round(dur, 3)
    except Exception:
        return None


def stage_data(cfg):
    from concurrent.futures import ProcessPoolExecutor

    train_manifest = f'{DATA}/metadata_train.csv'
    if os.path.exists(train_manifest) and glob.glob(f'{DATA}/wavs/*.wav'):
        print('Processed data already present, skipping data stage.')
        return

    meta_path, wav_dir = _locate_dataset(cfg)

    # metadata.csv: file_id | romanized | sinhala | speaker   (pipe, no header)
    rows = []
    with open(meta_path, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('|')
            if len(p) == 4 and p[3] == cfg.speaker:
                text = _clean_text(p[2])
                if text:
                    rows.append((p[0], text))
    print(f'{len(rows)} transcripts for speaker "{cfg.speaker}"')

    rows = [(i, t) for i, t in rows if os.path.exists(f'{wav_dir}/{i}.wav')]
    random.seed(SEED)
    random.shuffle(rows)
    if cfg.smoke:
        rows = rows[:cfg.smoke_clips]
    print(f'{len(rows)} clips to process')

    os.makedirs(f'{DATA}/wavs', exist_ok=True)
    jobs = [(f'{wav_dir}/{i}.wav', f'{DATA}/wavs/{i}.wav') for i, _ in rows]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as ex:
        durs = list(ex.map(process_clip, jobs, chunksize=16))

    kept = [(i, t, d) for (i, t), d in zip(rows, durs) if d is not None]
    hours = sum(d for _, _, d in kept) / 3600
    print(f'kept {len(kept)}/{len(rows)} clips | {hours:.2f} h usable audio')

    n_eval = cfg.eval_clips if cfg.smoke else min(200, max(24, int(0.02 * len(kept))))
    with open(f'{DATA}/metadata_eval.csv', 'w', encoding='utf-8') as fe, \
         open(train_manifest, 'w', encoding='utf-8') as ft:
        for k, (i, t, _) in enumerate(kept):
            (fe if k < n_eval else ft).write(f'{i}|{t}|{t}\n')
    print(f'train: {len(kept) - n_eval} | eval: {n_eval}')

    os.makedirs(WORK, exist_ok=True)
    json.dump({'speaker': cfg.speaker, 'clips': len(kept), 'hours': round(hours, 2),
               'train': len(kept) - n_eval, 'eval': n_eval, 'smoke': cfg.smoke},
              open(f'{WORK}/data_stats.json', 'w'))


# ----------------------------------------------------------------------------
# stage: train
# ----------------------------------------------------------------------------
def _find_resume():
    """checkpoint in working dir (same session) -> continue; in an attached
    previous-version output (/kaggle/input) -> restore weights into a new run."""
    for run in sorted(glob.glob(f'{OUT}/vits_sinhala*'), reverse=True):
        if glob.glob(f'{run}/*.pth'):
            return 'continue', run
    if os.path.exists('/kaggle/input'):
        cand = glob.glob('/kaggle/input/*/**/checkpoint_*.pth', recursive=True) \
             + glob.glob('/kaggle/input/*/**/best_model.pth', recursive=True)
        if cand:
            return 'restore', max(cand, key=os.path.getmtime)
    return None, None


def stage_train(cfg):
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits, VitsAudioConfig
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    # torchaudio.info fallback (some wav readers fail on it; harmless if unused)
    import torchaudio
    from TTS.tts.datasets import dataset as tts_dataset

    def _get_audio_size(path):
        try:
            return torchaudio.info(path).num_frames
        except Exception:
            import librosa
            return len(librosa.load(path, sr=None)[0])
    tts_dataset.get_audio_size = _get_audio_size

    # bf16-safe dashboard logging: CPU autocast emits bfloat16, which numpy
    # cannot convert -> the trainer's spectrogram plot crashes mid-training.
    # Cast tensors to float32 before plotting (harmless on GPU fp16 too).
    import TTS.tts.models.vits as _vits_mod
    _orig_plot_results = _vits_mod.plot_results

    def _safe_plot_results(y_hat, y, *args, **kwargs):
        try:
            y_hat = y_hat.float()
        except Exception:
            pass
        try:
            y = y.float()
        except Exception:
            pass
        return _orig_plot_results(y_hat, y, *args, **kwargs)
    _vits_mod.plot_results = _safe_plot_results

    os.makedirs(OUT, exist_ok=True)
    train_ds = BaseDatasetConfig(formatter='ljspeech', meta_file_train='metadata_train.csv', path=DATA)
    eval_ds = BaseDatasetConfig(formatter='ljspeech', meta_file_train='metadata_eval.csv', path=DATA)

    # vocabulary from the manifests (restart-proof, deterministic across sessions)
    texts = []
    for mf in ('metadata_train.csv', 'metadata_eval.csv'):
        with open(f'{DATA}/{mf}', encoding='utf-8') as f:
            texts += [ln.split('|')[1] for ln in f if ln.count('|') >= 2]
    letters = ''.join(sorted(set(''.join(texts)) - set(PUNCT)))
    print(f'vocab: {len(letters)} letters + {len(PUNCT)} punctuation marks')

    audio = VitsAudioConfig(sample_rate=SR, win_length=1024, hop_length=256,
                            num_mels=80, mel_fmin=0, mel_fmax=None)

    text_kwargs = dict(use_phonemes=False,
                       characters=CharactersConfig(
                           characters_class='TTS.tts.utils.text.characters.Graphemes',
                           pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>',
                           characters=letters, punctuations=PUNCT))
    if cfg.phonemes:  # experiment E2: espeak-ng Sinhala phoneme input
        text_kwargs = dict(use_phonemes=True, phonemizer='espeak', phoneme_language='si',
                           phoneme_cache_path=f'{TMP}/phoneme_cache')

    config = VitsConfig(
        audio=audio,
        run_name='vits_sinhala',
        batch_size=cfg.batch,
        eval_batch_size=8,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        epochs=cfg.epochs,
        text_cleaner='multilingual_cleaners',
        mixed_precision=True,
        output_path=OUT,
        datasets=[train_ds],
        test_sentences=TEST_SENTENCES,
        test_delay_epochs=1 if cfg.smoke else 10,
        print_step=5 if cfg.smoke else 50,
        save_step=20 if cfg.smoke else 2000,
        save_n_checkpoints=2,
        save_best_after=0 if cfg.smoke else 5000,
        **text_kwargs,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, _ = load_tts_samples(train_ds, eval_split=False)
    eval_samples, _ = load_tts_samples(eval_ds, eval_split=False)
    steps_per_epoch = max(1, len(train_samples) // cfg.batch)
    print(f'train {len(train_samples)} | eval {len(eval_samples)} | '
          f'~{steps_per_epoch} steps/epoch -> {cfg.epochs} epochs = {steps_per_epoch * cfg.epochs} steps max')

    targs = TrainerArgs()
    mode, path = _find_resume()
    if mode == 'continue':
        targs.continue_path = path
        print(f'RESUMING run in-place: {path}')
    elif mode == 'restore':
        targs.restore_path = path
        print(f'RESTORING weights from previous version: {path}')
    else:
        print('Fresh training run.')

    model = Vits(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(targs, config, OUT, model=model,
                      train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()


# ----------------------------------------------------------------------------
# stage: infer / package
# ----------------------------------------------------------------------------
def _latest_run():
    runs = [r for r in sorted(glob.glob(f'{OUT}/vits_sinhala*'), reverse=True)
            if glob.glob(f'{r}/*.pth')]
    assert runs, f'no trained run with checkpoints found under {OUT}'
    run = runs[0]
    best = f'{run}/best_model.pth'
    if os.path.exists(best):
        return run, best
    ckpts = sorted(glob.glob(f'{run}/checkpoint_*.pth'),
                   key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))
    return run, ckpts[-1]


def stage_infer(cfg):
    import torch
    from TTS.utils.synthesizer import Synthesizer
    run, ckpt = _latest_run()
    print('Synthesizing with', ckpt)
    syn = Synthesizer(tts_checkpoint=ckpt, tts_config_path=f'{run}/config.json',
                      use_cuda=torch.cuda.is_available())
    os.makedirs(f'{WORK}/samples', exist_ok=True)
    for k, sent in enumerate(TEST_SENTENCES):
        wav = syn.tts(sent)
        syn.save_wav(wav, f'{WORK}/samples/sample_{k}.wav')
        print(f'  samples/sample_{k}.wav <- {sent}')


def stage_package(cfg):
    run, ckpt = _latest_run()
    os.makedirs(EXPORT, exist_ok=True)
    for src in [ckpt, f'{run}/config.json', f'{WORK}/data_stats.json']:
        if os.path.exists(src):
            shutil.copy(src, EXPORT)
    for mf in glob.glob(f'{DATA}/metadata_*.csv'):
        shutil.copy(mf, EXPORT)
    print('Export contents:', os.listdir(EXPORT))
    print('\nNow: File > Save Version > Quick Save. Next session, attach this '
          'version output as an Input and rerun --stage all to resume.')


# ----------------------------------------------------------------------------
STAGES = {'setup': stage_setup, 'data': stage_data, 'train': stage_train,
          'infer': stage_infer, 'package': stage_package}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', default='all', choices=list(STAGES) + ['all'])
    p.add_argument('--smoke', action='store_true', help='tiny end-to-end verification run')
    p.add_argument('--phonemes', action='store_true', help='espeak-ng Sinhala phonemes (E2)')
    p.add_argument('--speaker', default='mettananda', choices=['mettananda', 'oshadi'])
    p.add_argument('--dataset-dir', default=None,
                   help='path to an already-present dataset (attached Kaggle Input or '
                        'kagglehub download); skips the GitHub download if it has metadata.csv + sinh_*.wav')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=None)
    cfg = p.parse_args()

    # Kaggle's "GPU T4 x2" exposes 2 GPUs; coqui-trainer refuses to auto-pick
    # and single-GPU is simplest/most robust here. Pin to GPU 0 unless the user
    # already set CUDA_VISIBLE_DEVICES (must happen before torch imports CUDA).
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    cfg.smoke_clips, cfg.eval_clips = 100, 8
    if cfg.epochs is None:
        cfg.epochs = 3 if cfg.smoke else 1000   # full: quota-limited, resume until good
    if cfg.batch is None:
        cfg.batch = 8 if cfg.smoke else 24      # 24 fits a 16 GB T4 with fp16

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
