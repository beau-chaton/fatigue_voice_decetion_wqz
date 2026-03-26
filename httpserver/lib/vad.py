"""
lib/vad.py
Silero VAD 本地离线加载与语音占比检测

包含：
  - _ensure_syspath_has        : 将路径插入 sys.path（去重）
  - speech_ratio_from_timestamps : 由时间戳列表计算语音占比
  - build_silero_vad_local     : 从本地 third_party 加载 Silero VAD 模型
  - detect_speech_ratio_silero : 对音频执行 VAD 并返回语音占比
"""

import sys
from pathlib import Path

import numpy as np
import torch

from lib.config import cfg


def _ensure_syspath_has(path: Path) -> None:
    """确保指定路径在 sys.path 中（避免重复添加）。"""
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def speech_ratio_from_timestamps(
    timestamps: list[dict], total_samples: int
) -> float:
    """
    根据 Silero VAD 返回的时间戳列表计算语音占总样本数的比例。

    Args:
        timestamps    : [{"start": int, "end": int}, ...]
        total_samples : 音频总样本数

    Returns:
        float in [0, 1]
    """
    if not timestamps:
        return 0.0
    speech_samples = sum(int(t["end"]) - int(t["start"]) for t in timestamps)
    return float(speech_samples) / float(total_samples)


def build_silero_vad_local():
    """
    从本地 third_party/silero-vad 加载 Silero VAD 模型（离线，不联网）。

    Returns:
        (model, get_speech_timestamps_fn)

    Raises:
        FileNotFoundError : 本地路径缺失时
        ImportError       : 无法导入 utils_vad 时
    """
    silero_repo_dir = cfg.silero_repo_dir
    silero_jit_path = cfg.silero_jit_path

    if not silero_repo_dir.exists():
        raise FileNotFoundError(
            f"Missing {silero_repo_dir} (expected silero-vad repo here)"
        )
    if not silero_jit_path.exists():
        raise FileNotFoundError(
            f"Missing {silero_jit_path} (export silero_vad.jit first)"
        )

    silero_src = silero_repo_dir / "src"
    if not silero_src.exists():
        raise FileNotFoundError(
            f"Missing {silero_src}. Your repo should have third_party/silero-vad/src/"
        )

    _ensure_syspath_has(silero_src)

    try:
        from silero_vad.utils_vad import get_speech_timestamps  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import get_speech_timestamps.\n"
            "Expected file at:\n"
            f"  {silero_src / 'silero_vad' / 'utils_vad.py'}"
        ) from e

    model = torch.jit.load(str(silero_jit_path))
    model.eval()
    return model, get_speech_timestamps


def detect_speech_ratio_silero(
    audio_f32: np.ndarray,
    sr: int,
    silero_model,
    get_speech_timestamps_fn,
) -> float:
    """
    对单声道 float32 音频执行 Silero VAD，返回语音占比。

    Args:
        audio_f32             : mono float32 ndarray
        sr                    : 采样率（通常 16000）
        silero_model          : build_silero_vad_local() 返回的模型
        get_speech_timestamps_fn : build_silero_vad_local() 返回的函数

    Returns:
        float in [0, 1]，语音帧占总帧的比例
    """
    wav = torch.from_numpy(audio_f32)
    with torch.no_grad():
        timestamps = get_speech_timestamps_fn(wav, silero_model, sampling_rate=sr)
    return speech_ratio_from_timestamps(timestamps, total_samples=len(audio_f32))
