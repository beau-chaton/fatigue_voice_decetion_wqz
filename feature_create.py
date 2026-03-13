import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import opensmile
import librosa
import soundfile as sf

DATA_DIR = Path(r"voice_data\openslr")  # 你的解压目录（Windows 用 r"" 更安全）
OUT_CSV = Path("features_egemaps.csv")

# 统一参数
TARGET_SR = 16000
WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SECONDS)

# 临时预处理输出目录（不改原始wav）
PREPROC_DIR = Path("tmp_preproc_3s_16k_mono")
PROGRESS_EVERY = 10


def label_from_folder(folder_name: str) -> int:
    return 1 if "Sleepy" in folder_name else 0


def speaker_from_folder(folder_name: str) -> str:
    return folder_name.split("_")[0]


def load_resample_mono(path: Path, target_sr: int) -> np.ndarray:
    # librosa 会自动转 mono（mono=True）
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    # 截断或补齐到固定长度
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        pad = target_len - len(y)
        return np.pad(y, (0, pad), mode="constant")
    return y


def preproc_to_tmp(wav_path: Path) -> Path:
    # 输出到 tmp_preproc_3s_16k_mono/原文件相对路径.wav
    rel = wav_path.relative_to(DATA_DIR)
    out_path = (PREPROC_DIR / rel).with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = load_resample_mono(wav_path, TARGET_SR)
    y = fix_length(y, WINDOW_SAMPLES)

    sf.write(str(out_path), y, TARGET_SR)
    return out_path


def main():
    start_all = time.perf_counter()
    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[START] scan wav under: {DATA_DIR.resolve()}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("[INIT] opensmile model ready")

    rows = []
    wav_files = sorted(DATA_DIR.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wav found under {DATA_DIR.resolve()}")

    total = len(wav_files)
    print(f"[INFO] found {total} wav files")

    for idx, wav in enumerate(wav_files, start=1):
        file_start = time.perf_counter()
        folder = wav.parent.name
        y_label = label_from_folder(folder)
        speaker = speaker_from_folder(folder)

        t0 = time.perf_counter()
        wav_3s = preproc_to_tmp(wav)
        t1 = time.perf_counter()

        feats = smile.process_file(str(wav_3s))
        t2 = time.perf_counter()
        row = feats.iloc[0].to_dict()
        row.update(
            {
                "path": str(wav),
                "path_3s": str(wav_3s),
                "speaker": speaker,
                "label_sleepy": y_label,
                "folder": folder,
                "sr": TARGET_SR,
                "window_seconds": WINDOW_SECONDS,
            }
        )
        rows.append(row)

        if idx <= 3 or idx % PROGRESS_EVERY == 0 or idx == total:
            elapsed = time.perf_counter() - file_start
            done = idx / total
            eta_sec = (time.perf_counter() - start_all) / done * (1 - done) if done > 0 else 0.0
            print(
                f"[PROGRESS] {idx}/{total} ({done:.1%}) | "
                f"prep={t1 - t0:.2f}s feat={t2 - t1:.2f}s total={elapsed:.2f}s | "
                f"ETA~{eta_sec:.1f}s | {wav.name}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    total_time = time.perf_counter() - start_all
    print(f"[DONE] total_time={total_time:.2f}s avg_per_file={total_time / total:.2f}s")
    print(f"Saved: {OUT_CSV} rows={len(df)} cols={len(df.columns)}")
    print(f"Preprocessed wavs written under: {PREPROC_DIR.resolve()}")


if __name__ == "__main__":
    main()