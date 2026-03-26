"""
lib/scoring.py
疲劳评分工具

包含：
  - audio_sanity_check  : 音频基础健壮性检查
  - three_state_weights : 根据 fatigue_score 计算三段状态权重
                          （energetic / normal / fatigue），和为 1
"""

import numpy as np


def audio_sanity_check(audio_f32: np.ndarray) -> tuple[bool, str]:
    """
    对输入音频做基础健壮性检查。

    Returns:
        (ok: bool, reason: str)  reason 在 ok=True 时为 "ok"
    """
    if audio_f32 is None:
        return False, "audio is None"
    if audio_f32.ndim != 1:
        return False, f"audio ndim != 1 (ndim={audio_f32.ndim})"
    if len(audio_f32) == 0:
        return False, "audio length = 0"
    if not np.isfinite(audio_f32).all():
        return False, "audio contains NaN/Inf"
    if np.max(np.abs(audio_f32)) == 0.0:
        return False, "audio is all zeros"
    return True, "ok"


def three_state_weights(
    fatigue_score: float,
    low: float | None = None,
    high: float | None = None,
    additive: float | None = None,
) -> dict:
    """
    根据 fatigue_score（P(Sleepy) 或其 EMA）计算三段状态权重。

    规则：
      - score < low       → energetic 最大
      - low <= score <= high → normal 最大
      - score > high      → fatigue 最大

    权重经 softmax 式归一化后和为 1，additive 避免某一权重归零。

    Args:
        fatigue_score : float in [0, 1]
        low           : energetic/normal 分界阈值（默认取 config.json）
        high          : normal/fatigue 分界阈值（默认取 config.json）
        additive      : 每个分量加上的基础偏置（默认取 config.json）

    Returns:
        dict with keys: energetic, normal, fatigue（均为 float，精确到 5 位小数）
    """
    from lib.config import cfg  # 延迟导入，避免循环依赖

    if low is None:
        low = cfg.energetic_threshold
    if high is None:
        high = cfg.fatigue_threshold
    if additive is None:
        additive = cfg.additive

    s = float(np.clip(fatigue_score, 0.0, 1.0))
    low = float(low)
    high = float(high)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Require 0<=low<high<=1")
    if additive < 0.0:
        raise ValueError("additive must be >= 0")

    if s < low:
        energetic = (low - s) / low       # 1 → 0
        normal = s / low                  # 0 → 1
        fatigue = 0.0
        normal = normal * normal          # 压 normal，保证 energetic 在整个 (0, low) 都最大

    elif s <= high:
        mid = (low + high) / 2.0
        if s <= mid:
            t = (s - low) / (mid - low + 1e-12)   # 0..1
            energetic = (1.0 - t)
            normal = 1.0
            fatigue = 0.0
            energetic = energetic * energetic
        else:
            t = (s - mid) / (high - mid + 1e-12)  # 0..1
            energetic = 0.0
            normal = 1.0
            fatigue = t
            fatigue = fatigue * fatigue

    else:
        t = (s - high) / (1.0 - high + 1e-12)     # 0..1
        energetic = 0.0
        normal = (1.0 - t)
        fatigue = 1.0
        normal = normal * normal

    if additive > 0.0:
        energetic += additive
        normal += additive
        fatigue += additive

    w_sum = energetic + normal + fatigue
    if w_sum <= 0:
        energetic, normal, fatigue = 0.0, 1.0, 0.0
        w_sum = 1.0

    energetic /= w_sum
    normal /= w_sum
    fatigue /= w_sum

    return {
        "energetic": round(float(energetic), 5),
        "normal": round(float(normal), 5),
        "fatigue": round(float(fatigue), 5),
    }
