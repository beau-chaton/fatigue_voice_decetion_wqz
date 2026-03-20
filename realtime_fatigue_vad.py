import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import joblib
import opensmile
import webrtcvad
import soundfile as sf
#检测是否有人说话，无人说话则不进行疲劳检测

# =========================
# Config
# =========================
MODEL_PATH = Path("sleepy_opensmile_logreg.joblib")

SR = 16000
CHANNELS = 1

WINDOW_SECONDS = 3.0           # 每次判定用 3 秒
HOP_SECONDS = 3.0              # 每隔多久更新一次（=3s 就是不重叠；改成 1s 就是重叠滑窗）
VAD_FRAME_MS = 30              # webrtcvad 支持 10/20/30 ms
VAD_MODE = 1                   # 0-3 越大越激进(更容易判成语音). 一般 2 或 3
MIN_SPEECH_RATIO = 0.35        # 3 秒窗口中，语音帧比例低于这个就认为“无有效语音”

EMA_ALPHA = 0.8                # 平滑系数：越大越跟随当前，越小越平滑
FREEZE_WHEN_NO_SPEECH = True   # 无语音时是否保持上一次分数（冻结）。否则输出 None

# 可选：调试保存最近一次窗口音频
DEBUG_SAVE_LAST_WAV = False
DEBUG_WAV_PATH = Path("debug_last_window.wav")


# =========================
# Helpers
# =========================
def float_to_pcm16(x: np.ndarray) -> bytes:
    """[-1,1] float32 -> PCM16 bytes"""
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    return pcm.tobytes()


def vad_speech_ratio(audio_f32: np.ndarray, sr: int, vad: webrtcvad.Vad, frame_ms: int) -> float:
    """Return ratio of frames detected as speech."""
    assert sr == 16000, "webrtcvad requires 16kHz for best compatibility"
    assert frame_ms in (10, 20, 30)

    frame_len = int(sr * frame_ms / 1000)
    if len(audio_f32) < frame_len:
        return 0.0

    pcm_bytes = float_to_pcm16(audio_f32)
    bytes_per_frame = frame_len * 2  # int16 = 2 bytes

    n_frames = len(pcm_bytes) // bytes_per_frame
    if n_frames == 0:
        return 0.0

    speech = 0
    for i in range(n_frames):
        frame = pcm_bytes[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        if vad.is_speech(frame, sr):
            speech += 1
    return speech / n_frames


def record_window(sr: int, seconds: float) -> np.ndarray:
    """Record mono float32 audio at sr for seconds."""
    n = int(sr * seconds)
    audio = sd.rec(n, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]  # shape (n,)


@dataclass
class FatigueEngine:
    model: object
    feature_cols: list[str]
    smile: opensmile.Smile
    vad: webrtcvad.Vad
    ema_score: float | None = None

    def predict_score(self, audio_f32: np.ndarray) -> tuple[float | None, float]:
        """
        Returns (fatigue_score_or_None, speech_ratio)
        fatigue_score is P(Sleepy) in [0,1]
        """
        speech_ratio = vad_speech_ratio(audio_f32, SR, self.vad, VAD_FRAME_MS)

        if speech_ratio < MIN_SPEECH_RATIO:
            # no valid speech
            if FREEZE_WHEN_NO_SPEECH:
                return self.ema_score, speech_ratio
            return None, speech_ratio

        # Feature extraction: openSMILE expects a file or signal.
        # The Python 'opensmile' package supports process_signal in recent versions.
        try:
            feats = self.smile.process_signal(audio_f32, SR)
        except Exception:
            # fallback: write temp wav then process_file
            tmp = Path("_tmp_window.wav")
            sf.write(str(tmp), audio_f32, SR)
            feats = self.smile.process_file(str(tmp))
            try:
                tmp.unlink()
            except Exception:
                pass

        # Align columns to training
        x = feats.reindex(columns=self.feature_cols)
        if x.isnull().any(axis=None):
            # 某些列缺失/版本差异时，最安全策略：缺失列填 0
            x = x.fillna(0.0)

        p_sleepy = float(self.model.predict_proba(x)[0, 1])

        # EMA smoothing
        if self.ema_score is None:
            self.ema_score = p_sleepy
        else:
            self.ema_score = (1 - EMA_ALPHA) * self.ema_score + EMA_ALPHA * p_sleepy
        return self.ema_score, speech_ratio


def load_engine(model_path: Path) -> FatigueEngine:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    vad = webrtcvad.Vad(VAD_MODE)

    return FatigueEngine(model=model, feature_cols=feature_cols, smile=smile, vad=vad)


def main():
    print("[INIT] loading model:", MODEL_PATH.resolve())
    engine = load_engine(MODEL_PATH)
    print("[READY] start realtime fatigue scoring")
    print(f"window={WINDOW_SECONDS}s hop={HOP_SECONDS}s sr={SR} vad_mode={VAD_MODE} min_speech_ratio={MIN_SPEECH_RATIO}")

    while True:
        t0 = time.time()
        audio = record_window(SR, WINDOW_SECONDS)

        if DEBUG_SAVE_LAST_WAV:
            sf.write(str(DEBUG_WAV_PATH), audio, SR)

        score, speech_ratio = engine.predict_score(audio)

        # 输出策略
        speaking = speech_ratio >= MIN_SPEECH_RATIO
        speech_label = "有人说话" if speaking else "无人说话"
        if not speaking:
            print(f"[{speech_label}] speech_ratio={speech_ratio:.2f} fatigue_score=-- fatigued=False")
        else:
            effective_score = score if score is not None else 0.0
            fatigued = effective_score >= 0.7  # 你后续用验证集调这个阈值
            print(f"[{speech_label}] speech_ratio={speech_ratio:.2f} fatigue_score={effective_score:.3f} fatigued={fatigued}")

        # 控制 hop（让每次循环间隔约等于 HOP_SECONDS）
        elapsed = time.time() - t0
        sleep_left = max(0.0, HOP_SECONDS - elapsed)
        time.sleep(sleep_left)


if __name__ == "__main__":
    main()