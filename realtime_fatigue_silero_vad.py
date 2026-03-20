"""
realtime_fatigue_silero_vad.py

把 Silero VAD（是否有人说话） + openSMILE 特征 + joblib 疲劳模型（P(Sleepy)=fatigue_score）合并到一个脚本里。
- 阶段1：Silero VAD 计算 speech_ratio，低于阈值则认为“无人说话”，不更新疲劳分数（可冻结上一次分数）
- 阶段2：仅在“有人说话”时，提 eGeMAPS Functionals 特征，用 joblib 模型输出 fatigue_score

依赖：
pip install torch torchaudio sounddevice numpy opensmile joblib soundfile pandas
"""


import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import joblib
import opensmile
import soundfile as sf


# =========================
# Config (保持你的参数不变)
# =========================
MODEL_PATH = Path("sleepy_opensmile_logreg.joblib")

SILERO_REPO_DIR = Path("third_party/silero-vad")
SILERO_JIT_PATH = Path("assets/silero_vad.jit")

SR = 16000
CHANNELS = 1

WINDOW_SECONDS = 3.0          # 每次录音窗口长度（秒）
HOP_SECONDS = 3.0             # 输出刷新间隔（秒）；=3 不重叠；=1 更实时（重叠）

MIN_SPEECH_RATIO = 0.12       # Silero VAD: speech_ratio >= 该阈值认为“有人说话”

EMA_ALPHA = 0.5               # 平滑系数：越大越跟随当前，越小越平滑（建议 0.1~0.3）
FREEZE_WHEN_NO_SPEECH = True  # 无语音时是否保持上一次分数（冻结）。否则输出 None

FATIGUED_THRESHOLD = 0.7      # 疲劳阈值（你后续根据验证集调）

DEBUG_SAVE_LAST_WAV = False
DEBUG_WAV_PATH = Path("debug_last_window_16k.wav")

# 缺失特征报警阈值（不影响模型，只影响日志）
MISSING_FEATURE_WARN_RATIO = 0.05  # 缺失列超过 5% 就报警


# =========================
# Audio record
# =========================
def record_window(sr: int, seconds: float) -> np.ndarray: 
    """Record mono float32 audio at sr for seconds."""
    n = int(sr * seconds)
    audio = sd.rec(n, samplerate=sr, channels=CHANNELS, dtype="float32")
    sd.wait()
    return audio[:, 0]  # (n,)


def audio_sanity_check(audio_f32: np.ndarray) -> tuple[bool, str]:
    """
    不改变任何阈值/参数，只做健壮性保护：
    - NaN/Inf
    - 长度不对
    - 全 0（常见于设备异常/权限/输入选择错误）
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


# =========================
# Silero VAD (speech detection) - offline local loader
# =========================
def _ensure_syspath_has(path: Path) -> None:
    """
    只插一次 sys.path，避免重复插入导致路径膨胀或 import 行为不可控。
    """
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def speech_ratio_from_timestamps(timestamps: list[dict], total_samples: int) -> float:
    """timestamps: [{'start': sample_idx, 'end': sample_idx}, ...]"""
    if not timestamps:
        return 0.0
    speech_samples = 0
    for t in timestamps:
        speech_samples += int(t["end"]) - int(t["start"])
    return float(speech_samples) / float(total_samples)


def build_silero_vad_local():
    """
    纯离线加载（适配 repo 的 src/ 结构）：
    - 模型：assets/silero_vad.jit（torch.jit.load）
    - 工具函数：third_party/silero-vad/src/silero_vad/utils_vad.py 里的 get_speech_timestamps
    """
    if not SILERO_REPO_DIR.exists():
        raise FileNotFoundError(f"Missing {SILERO_REPO_DIR} (expected silero-vad repo here)")
    if not SILERO_JIT_PATH.exists():
        raise FileNotFoundError(f"Missing {SILERO_JIT_PATH} (export silero_vad.jit first)")

    silero_src = SILERO_REPO_DIR / "src"
    if not silero_src.exists():
        raise FileNotFoundError(f"Missing {silero_src}. Your repo should have third_party/silero-vad/src/")

    # 关键：把 src/ 加到 sys.path，使 silero_vad 成为顶层包（且只插一次）
    _ensure_syspath_has(silero_src)

    try:
        from silero_vad.utils_vad import get_speech_timestamps  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import get_speech_timestamps.\n"
            "Expected file at:\n"
            f"  {silero_src / 'silero_vad' / 'utils_vad.py'}"
        ) from e

    model = torch.jit.load(str(SILERO_JIT_PATH))
    model.eval()
    return model, get_speech_timestamps


def detect_speech_ratio_silero(
    audio_f32: np.ndarray,
    sr: int,
    silero_model,
    get_speech_timestamps_fn,
) -> float:
    """
    输入：单声道 float32 音频（-1~1）
    输出：speech_ratio（0~1）
    """
    wav = torch.from_numpy(audio_f32)

    # 推理阶段：不需要��度
    with torch.no_grad():
        timestamps = get_speech_timestamps_fn(wav, silero_model, sampling_rate=sr)

    return speech_ratio_from_timestamps(timestamps, total_samples=len(audio_f32))


# =========================
# Fatigue engine
# =========================
@dataclass
class FatigueEngine:
    model: object
    feature_cols: list[str]
    smile: opensmile.Smile
    silero_model: object
    get_speech_timestamps_fn: object
    ema_score: float | None = None
    _warned_missing_features: bool = False  # 只警告一次，避免刷屏

    def _align_features_with_warning(self, feats) -> tuple[object, int]:
        """
        对齐训练特征列，并对缺失列做一次性报警。
        返回：x, missing_count
        """
        # reindex 会把缺失列变成 NaN
        x = feats.reindex(columns=self.feature_cols)

        missing_mask = x.isnull().any(axis=0)
        missing_count = int(missing_mask.sum())

        # 只做日志报警，不改变你的模型参数/阈值
        if missing_count > 0 and not self._warned_missing_features:
            missing_ratio = missing_count / max(1, len(self.feature_cols))
            if missing_ratio >= MISSING_FEATURE_WARN_RATIO:
                missing_cols = list(x.columns[missing_mask])[:20]  # 最多展示 20 个，避免太长
                print(
                    "[WARN] openSMILE feature columns mismatch.\n"
                    f"       missing_count={missing_count}/{len(self.feature_cols)} ({missing_ratio:.1%})\n"
                    f"       examples={missing_cols}\n"
                    "       (Will fill NaN with 0.0 to continue, but outputs may be degraded.)"
                )
                self._warned_missing_features = True

        if x.isnull().any(axis=None):
            x = x.fillna(0.0)

        return x, missing_count

    def predict(self, audio_f32: np.ndarray) -> dict:
        """
        返回一个 dict，便于你后续接 UI / 日志：
        - speaking: bool
        - speech_ratio: float
        - fatigue_score: float | None  (None 表示没更新)
        - fatigue_score_raw: float | None
        - fatigued: bool
        """
        # 6) 输入音频 sanity check（不改变阈值，只在异常时保护）
        ok, reason = audio_sanity_check(audio_f32)
        if not ok:
            score = self.ema_score if FREEZE_WHEN_NO_SPEECH else None
            return {
                "speaking": False,
                "speech_ratio": 0.0,
                "fatigue_score_raw": None,
                "fatigue_score": score,
                "fatigued": bool(score is not None and score >= FATIGUED_THRESHOLD),
                "note": f"audio_invalid: {reason}",
            }

        speech_ratio = detect_speech_ratio_silero(
            audio_f32, SR, self.silero_model, self.get_speech_timestamps_fn
        )
        speaking = speech_ratio >= MIN_SPEECH_RATIO

        if not speaking:
            # 无人说话：不做疲劳推理
            score = self.ema_score if FREEZE_WHEN_NO_SPEECH else None
            return {
                "speaking": False,
                "speech_ratio": speech_ratio,
                "fatigue_score_raw": None,
                "fatigue_score": score,
                "fatigued": bool(score is not None and score >= FATIGUED_THRESHOLD),
            }

        # 有人说话：提特征 -> 模型出分数
        # 优先用 process_signal（更快，不用写临时文件）
        feats = None
        try:
            feats = self.smile.process_signal(audio_f32, SR)
        except Exception:
            # 4) 临时文件：避免固定名字冲突
            with tempfile.NamedTemporaryFile(prefix="opensmile_", suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)
            try:
                sf.write(str(tmp_path), audio_f32, SR)
                feats = self.smile.process_file(str(tmp_path))
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)  # py3.8+ on windows ok
                except Exception:
                    pass

        x, _missing_count = self._align_features_with_warning(feats)

        raw = float(self.model.predict_proba(x)[0, 1])  # P(Sleepy)

        # EMA 平滑（参数不改）
        if self.ema_score is None:
            self.ema_score = raw
        else:
            self.ema_score = (1 - EMA_ALPHA) * self.ema_score + EMA_ALPHA * raw

        score = self.ema_score
        fatigued = score >= FATIGUED_THRESHOLD

        return {
            "speaking": True,
            "speech_ratio": speech_ratio,
            "fatigue_score_raw": raw,
            "fatigue_score": score,
            "fatigued": fatigued,
        }


def load_engine(model_path: Path) -> FatigueEngine:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    silero_model, get_speech_timestamps_fn = build_silero_vad_local()

    return FatigueEngine(
        model=model,
        feature_cols=feature_cols,
        smile=smile,
        silero_model=silero_model,
        get_speech_timestamps_fn=get_speech_timestamps_fn,
    )


# =========================
# Main loop
# =========================
def main():
    print("[INIT] loading fatigue model:", MODEL_PATH.resolve())
    engine = load_engine(MODEL_PATH)
    print("[READY] realtime fatigue scoring + silero VAD")
    print(
        f"[CONFIG] sr={SR} window={WINDOW_SECONDS}s hop={HOP_SECONDS}s "
        f"min_speech_ratio={MIN_SPEECH_RATIO} ema_alpha={EMA_ALPHA} "
        f"freeze_no_speech={FREEZE_WHEN_NO_SPEECH}"
    )

    while True:
        t0 = time.time()
        audio = record_window(SR, WINDOW_SECONDS)

        if DEBUG_SAVE_LAST_WAV:
            sf.write(str(DEBUG_WAV_PATH), audio, SR)

        out = engine.predict(audio)

        note = out.get("note", "")
        note_suffix = f" | {note}" if note else ""

        if not out["speaking"]:
            score_str = "--" if out["fatigue_score"] is None else f'{out["fatigue_score"]:.3f}'
            print(
                f"[无人说话] speech_ratio={out['speech_ratio']:.3f} "
                f"fatigue_score={score_str} fatigued={out['fatigued']}{note_suffix}"
            )
        else:
            print(
                f"[有人说话] speech_ratio={out['speech_ratio']:.3f} "
                f"real_time_fatigue_score={out['fatigue_score_raw']:.3f} fatigue_score={out['fatigue_score']:.3f} "
                f"fatigued={out['fatigued']}{note_suffix}"
            )

        elapsed = time.time() - t0
        time.sleep(max(0.0, HOP_SECONDS - elapsed))


if __name__ == "__main__":
    main()