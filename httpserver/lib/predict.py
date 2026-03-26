"""
lib/predict.py
核心预测逻辑

包含：
  - _SESSION_EMA          : 进程级 session EMA 状态字典
  - _get_sid              : 规范化 session_id
  - _align_features_with_warning : 对齐特征列并补零
  - predict_audio         : 对 mono float32 音频执行疲劳预测（核心）
  - predict_from_source   : 从 wav_path / wav_url 加载并预测（对外接口）

配置常量（与原文件保持一致）：
  SR                     = 16000
  MIN_SPEECH_RATIO       = 0.12
  EMA_ALPHA              = 0.5
  FREEZE_WHEN_NO_SPEECH  = True
  FATIGUED_THRESHOLD     = 0.7
  ENERGETIC_THRESHOLD    = 0.4
  FATIGUE_THRESHOLD      = 0.7
  MISSING_FEATURE_WARN_RATIO = 0.05
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from lib.audio_io import SR, _resample, load_wav_from_path, load_wav_from_url
from lib.model import GlobalComponents, get_global_components
from lib.scoring import audio_sanity_check, three_state_weights
from lib.vad import detect_speech_ratio_silero

# ─── 运行时配置 ────────────────────────────────────────────────
MIN_SPEECH_RATIO = 0.12          # 低于此值视为无语音
EMA_ALPHA = 0.5                  # EMA 平滑系数
FREEZE_WHEN_NO_SPEECH = True     # 无语音时保持上一次 EMA 分数
FATIGUED_THRESHOLD = 0.7         # 判定为疲劳的阈值
ENERGETIC_THRESHOLD = 0.4        # 三段阈值：精力充沛上限
FATIGUE_THRESHOLD = 0.7          # 三段阈值：疲劳下限
MISSING_FEATURE_WARN_RATIO = 0.05  # 特征缺失比例超过此值时打印警告

# ─── Session EMA 状态 ─────────────────────────────────────────
_SESSION_EMA: dict[str, float | None] = {}


def _get_sid(session_id: str | None) -> str:
    """规范化 session_id，空值统一返回 'default'。"""
    sid = (session_id or "default").strip()
    return sid if sid else "default"


def _align_features_with_warning(global_: GlobalComponents, feats: Any) -> Any:
    """
    将 opensmile 输出的特征 DataFrame 按训练时的列顺序对齐。
    缺失列填 0.0，并在缺失比例超过阈值时打印一次警告。
    """
    x = feats.reindex(columns=global_.feature_cols)

    missing_mask = x.isnull().any(axis=0)
    missing_count = int(missing_mask.sum())

    if missing_count > 0 and not global_.warned_missing_features:
        missing_ratio = missing_count / max(1, len(global_.feature_cols))
        if missing_ratio >= MISSING_FEATURE_WARN_RATIO:
            missing_cols = list(x.columns[missing_mask])[:20]
            print(
                "[WARN] openSMILE feature columns mismatch.\n"
                f"       missing_count={missing_count}/{len(global_.feature_cols)}"
                f" ({missing_ratio:.1%})\n"
                f"       examples={missing_cols}\n"
                "       (Will fill NaN with 0.0 to continue,"
                " but outputs may be degraded.)"
            )
            global_.warned_missing_features = True

    if x.isnull().any(axis=None):
        x = x.fillna(0.0)

    return x


def predict_audio(
    audio_f32: np.ndarray,
    sr: int,
    session_id: str | None = None,
) -> dict:
    """
    对单声道 float32 音频执行疲劳度预测。

    流程：
      1. 音频健壮性检查
      2. Silero VAD → 语音占比
      3. opensmile 特征提取 → sklearn 模型推理
      4. EMA 平滑 → 三段状态权重

    Args:
        audio_f32  : mono float32 ndarray（已重采样到目标 SR）
        sr         : 音频采样率
        session_id : 用于跨帧 EMA 状态的会话标识

    Returns:
        dict: speaking, speech_ratio, fatigue_score_raw,
              fatigue_score, fatigued, state_weights, note
    """
    global_ = get_global_components()
    sid = _get_sid(session_id)

    # 1. 音频健壮性检查
    ok, reason = audio_sanity_check(audio_f32)
    if not ok:
        ema = _SESSION_EMA.get(sid, None)
        score = ema if FREEZE_WHEN_NO_SPEECH else None
        weights = None if score is None else three_state_weights(score)
        return {
            "speaking": False,
            "speech_ratio": 0.0,
            "fatigue_score_raw": None,
            "fatigue_score": score,
            "fatigued": bool(score is not None and score >= FATIGUED_THRESHOLD),
            "state_weights": weights,
            "note": f"audio_invalid: {reason}",
        }

    # 2. VAD → 语音占比
    speech_ratio = detect_speech_ratio_silero(
        audio_f32, sr, global_.silero_model, global_.get_speech_timestamps_fn
    )
    speaking = speech_ratio >= MIN_SPEECH_RATIO

    if not speaking:
        ema = _SESSION_EMA.get(sid, None)
        score = ema if FREEZE_WHEN_NO_SPEECH else None
        weights = None if score is None else three_state_weights(score)
        return {
            "speaking": False,
            "speech_ratio": float(speech_ratio),
            "fatigue_score_raw": None,
            "fatigue_score": score,
            "fatigued": bool(score is not None and score >= FATIGUED_THRESHOLD),
            "state_weights": weights,
            "note": "",
        }

    # 3. opensmile 特征提取
    feats = None
    try:
        feats = global_.smile.process_signal(audio_f32, sr)
    except Exception:
        # 回退：写临时文件再用 process_file
        with tempfile.NamedTemporaryFile(
            prefix="opensmile_", suffix=".wav", delete=False
        ) as f:
            tmp_path = Path(f.name)
        try:
            sf.write(str(tmp_path), audio_f32, sr)
            feats = global_.smile.process_file(str(tmp_path))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # 4. 特征对齐 → 推理 → EMA → 状态权重
    x = _align_features_with_warning(global_, feats)
    raw = float(global_.model.predict_proba(x.values)[0, 1])

    prev = _SESSION_EMA.get(sid, None)
    ema = raw if prev is None else (1 - EMA_ALPHA) * prev + EMA_ALPHA * raw
    _SESSION_EMA[sid] = ema

    fatigued = ema >= FATIGUED_THRESHOLD
    weights = three_state_weights(ema, additive=0.2)

    return {
        "speaking": True,
        "speech_ratio": float(speech_ratio),
        "fatigue_score_raw": raw,
        "fatigue_score": float(ema),
        "fatigued": bool(fatigued),
        "state_weights": weights,
        "note": "",
    }


def predict_from_source(
    wav_path: str | None = None,
    wav_url: str | None = None,
    session_id: str | None = None,
    options: dict | None = None,
) -> dict:
    """
    从文件路径或 URL 加载音频并执行疲劳度预测。

    Args:
        wav_path   : 本地 WAV 文件路径（与 wav_url 二选一）
        wav_url    : 远程 WAV 文件 URL（与 wav_path 二选一）
        session_id : 会话标识（用于 EMA 平滑）
        options    : 可选参数字典
            - resample_to_16k (bool, 默认 True)
            - timeout_sec     (float, 默认 10.0，仅 URL 模式有效)

    Returns:
        dict: ok, session_id, input{...}, output{...}
    """
    options = options or {}
    resample_to_16k = bool(options.get("resample_to_16k", True))
    timeout_sec = float(options.get("timeout_sec", 10.0))

    if wav_path:
        audio, sr_in, ch_in, dur = load_wav_from_path(wav_path)
        src_kind = "path"
        src_value = wav_path
    elif wav_url:
        audio, sr_in, ch_in, dur = load_wav_from_url(wav_url, timeout_sec=timeout_sec)
        src_kind = "url"
        src_value = wav_url
    else:
        raise ValueError("Either wav_path or wav_url must be provided")

    sr_used = sr_in
    if resample_to_16k:
        audio = _resample(audio, sr_in=sr_in, sr_out=SR)
        sr_used = SR

    core = predict_audio(audio_f32=audio, sr=sr_used, session_id=session_id)

    return {
        "ok": True,
        "session_id": _get_sid(session_id),
        "input": {
            "source_kind": src_kind,
            "source_value": src_value,
            "sr_in": sr_in,
            "sr_used": sr_used,
            "channels_in": ch_in,
            "duration_sec": dur,
        },
        "output": core,
    }


def session_ema_reset(session_id: str | None) -> tuple[str, bool]:
    """
    清除指定 session 的 EMA 状态。

    Returns:
        (sid: str, existed: bool)
    """
    sid = _get_sid(session_id)
    existed = sid in _SESSION_EMA
    _SESSION_EMA.pop(sid, None)
    return sid, existed
