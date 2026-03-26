"""
lib/audio_io.py
音频加载与格式转换工具

包含：
  - _to_float32      : 将任意 dtype 的 ndarray 转为 float32
  - _ensure_mono     : 将多声道音频混为单声道
  - _resample        : 使用 resample_poly 重采样
  - load_wav_from_path : 从文件路径加载 wav
  - load_wav_from_url  : 从 HTTP URL 下载并加载 wav
"""

import io
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

# 目标采样率（全局常量，与 config 保持一致）
SR = 16000


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """将任意数值类型的音频数组转换为 float32，归一化到 [-1, 1]。"""
    if audio.dtype == np.float32:
        return audio
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32, copy=False)

    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0

    return audio.astype(np.float32)


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """将多声道音频平均混为单声道；已是单声道则直接返回。"""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"Unsupported audio ndim: {audio.ndim}")


def _resample(audio_f32: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """使用 scipy.signal.resample_poly 对音频进行重采样。"""
    if sr_in == sr_out:
        return audio_f32
    if sr_in <= 0:
        raise ValueError(f"Invalid sr_in={sr_in}")

    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(audio_f32, up, down).astype(np.float32)


def load_wav_from_path(wav_path: str) -> tuple[np.ndarray, int, int, float]:
    """
    从文件路径加载 WAV 文件。

    Returns:
        (audio_mono_f32, sr_in, channels_in, duration_sec)
    """
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"wav_path not found: {wav_path}")

    audio, sr_in = sf.read(str(p), always_2d=True)  # (n, c)
    channels_in = int(audio.shape[1])
    audio = _ensure_mono(_to_float32(audio))
    duration_sec = float(len(audio)) / float(sr_in) if sr_in > 0 else 0.0
    return audio, int(sr_in), channels_in, duration_sec


def load_wav_from_url(
    wav_url: str, timeout_sec: float = 10.0
) -> tuple[np.ndarray, int, int, float]:
    """
    通过 HTTP URL 下载并加载 WAV 文件。

    Returns:
        (audio_mono_f32, sr_in, channels_in, duration_sec)
    """
    if not wav_url:
        raise ValueError("wav_url is empty")

    r = requests.get(wav_url, timeout=timeout_sec)
    r.raise_for_status()

    bio = io.BytesIO(r.content)
    audio, sr_in = sf.read(bio, always_2d=True)
    channels_in = int(audio.shape[1])
    audio = _ensure_mono(_to_float32(audio))
    duration_sec = float(len(audio)) / float(sr_in) if sr_in > 0 else 0.0
    return audio, int(sr_in), channels_in, duration_sec
