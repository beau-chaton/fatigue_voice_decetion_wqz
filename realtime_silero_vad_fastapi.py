import io
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

import numpy as np

import torch
import joblib
import opensmile
import soundfile as sf
from scipy.signal import resample_poly
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel


# =========================
# Config (保持你的参数不变)
# =========================
MODEL_PATH = Path("sleepy_opensmile_logreg.joblib")

SILERO_REPO_DIR = Path("third_party/silero-vad")
SILERO_JIT_PATH = Path("assets/silero_vad.jit")

SR = 16000

MIN_SPEECH_RATIO = 0.12

EMA_ALPHA = 0.5
FREEZE_WHEN_NO_SPEECH = True

FATIGUED_THRESHOLD = 0.7

# 你提出的三段阈值
ENERGETIC_THRESHOLD = 0.4
FATIGUE_THRESHOLD = 0.7

MISSING_FEATURE_WARN_RATIO = 0.05


def audio_sanity_check(audio_f32: np.ndarray) -> tuple[bool, str]:
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
# Audio IO (API)
# =========================
def _to_float32(audio: np.ndarray) -> np.ndarray:
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
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"Unsupported audio ndim: {audio.ndim}")


def _resample(audio_f32: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return audio_f32
    if sr_in <= 0:
        raise ValueError(f"Invalid sr_in={sr_in}")

    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = resample_poly(audio_f32, up, down).astype(np.float32)
    return y


def load_wav_from_path(wav_path: str) -> tuple[np.ndarray, int, int, float]:
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"wav_path not found: {wav_path}")

    audio, sr_in = sf.read(str(p), always_2d=True)  # (n, c)
    channels_in = int(audio.shape[1])
    audio = _ensure_mono(_to_float32(audio))
    duration_sec = float(len(audio)) / float(sr_in) if sr_in > 0 else 0.0
    return audio, int(sr_in), channels_in, duration_sec


def load_wav_from_url(wav_url: str, timeout_sec: float = 10.0) -> tuple[np.ndarray, int, int, float]:
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


# =========================
# 3-state weights
# =========================
def three_state_weights(
    fatigue_score: float,
    low: float = ENERGETIC_THRESHOLD,
    high: float = FATIGUE_THRESHOLD,
) -> dict:
    """
    输入 fatigue_score in [0,1]（P(Sleepy) or EMA of it）
    输出三个权重：energetic/normal/fatigue，且和为1，并确保“当前状态”权重最大。

    规则：
    - score < low     -> energetic 最大
    - low<=score<=high-> normal 最大
    - score > high    -> fatigue 最大
    """
    s = float(np.clip(fatigue_score, 0.0, 1.0))
    low = float(low)
    high = float(high)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Require 0<=low<high<=1")

    if s < low:
        energetic = (low - s) / low  # 1 -> 0
        normal = s / low  # 0 -> 1
        fatigue = 0.0
        # 压 normal，保证 energetic 在整个 (0, low) 都最大
        normal = normal * normal

    elif s <= high:
        # normal 最大区间
        mid = (low + high) / 2.0
        if s <= mid:
            # energetic -> normal
            t = (s - low) / (mid - low + 1e-12)  # 0..1
            energetic = (1.0 - t)
            normal = 1.0
            fatigue = 0.0
            energetic = energetic * energetic
        else:
            # normal -> fatigue
            t = (s - mid) / (high - mid + 1e-12)  # 0..1
            energetic = 0.0
            normal = 1.0
            fatigue = t
            fatigue = fatigue * fatigue

    else:
        # fatigue 最大区间
        t = (s - high) / (1.0 - high + 1e-12)  # 0..1
        energetic = 0.0
        normal = (1.0 - t)
        fatigue = 1.0
        normal = normal * normal

    w_sum = energetic + normal + fatigue
    if w_sum <= 0:
        energetic, normal, fatigue = 0.0, 1.0, 0.0
        w_sum = 1.0

    energetic /= w_sum
    normal /= w_sum
    fatigue /= w_sum

    return {
        "energetic": round(float(energetic), 2),
        "normal": round(float(normal), 2),
        "fatigue": round(float(fatigue), 2),
    }


# =========================
# Silero VAD loader (offline local)
# =========================
def _ensure_syspath_has(path: Path) -> None:
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def speech_ratio_from_timestamps(timestamps: list[dict], total_samples: int) -> float:
    if not timestamps:
        return 0.0
    speech_samples = 0
    for t in timestamps:
        speech_samples += int(t["end"]) - int(t["start"])
    return float(speech_samples) / float(total_samples)


def build_silero_vad_local():
    if not SILERO_REPO_DIR.exists():
        raise FileNotFoundError(f"Missing {SILERO_REPO_DIR} (expected silero-vad repo here)")
    if not SILERO_JIT_PATH.exists():
        raise FileNotFoundError(f"Missing {SILERO_JIT_PATH} (export silero_vad.jit first)")

    silero_src = SILERO_REPO_DIR / "src"
    if not silero_src.exists():
        raise FileNotFoundError(f"Missing {silero_src}. Your repo should have third_party/silero-vad/src/")

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
    wav = torch.from_numpy(audio_f32)
    with torch.no_grad():
        timestamps = get_speech_timestamps_fn(wav, silero_model, sampling_rate=sr)
    return speech_ratio_from_timestamps(timestamps, total_samples=len(audio_f32))


# =========================
# Global components (load once)
# =========================
@dataclass
class GlobalComponents:
    model: Any
    feature_cols: list[str]
    smile: opensmile.Smile
    silero_model: Any
    get_speech_timestamps_fn: Any
    warned_missing_features: bool = False


_GLOBAL: GlobalComponents | None = None


def get_global_components() -> GlobalComponents:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    silero_model, get_speech_timestamps_fn = build_silero_vad_local()

    _GLOBAL = GlobalComponents(
        model=model,
        feature_cols=feature_cols,
        smile=smile,
        silero_model=silero_model,
        get_speech_timestamps_fn=get_speech_timestamps_fn,
    )
    return _GLOBAL


# =========================
# Session EMA store (only float)
# =========================
_SESSION_EMA: dict[str, float | None] = {}


def _get_sid(session_id: str | None) -> str:
    sid = (session_id or "default").strip()
    return sid if sid else "default"


# =========================
# Core predict (stateless except ema float)
# =========================
def _align_features_with_warning(global_: GlobalComponents, feats) -> Any:
    x = feats.reindex(columns=global_.feature_cols)

    missing_mask = x.isnull().any(axis=0)
    missing_count = int(missing_mask.sum())

    if missing_count > 0 and not global_.warned_missing_features:
        missing_ratio = missing_count / max(1, len(global_.feature_cols))
        if missing_ratio >= MISSING_FEATURE_WARN_RATIO:
            missing_cols = list(x.columns[missing_mask])[:20]
            print(
                "[WARN] openSMILE feature columns mismatch.\n"
                f"       missing_count={missing_count}/{len(global_.feature_cols)} ({missing_ratio:.1%})\n"
                f"       examples={missing_cols}\n"
                "       (Will fill NaN with 0.0 to continue, but outputs may be degraded.)"
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
    输入必须是 mono float32.
    返回 dict（API 会直接包装它）。
    """
    global_ = get_global_components()
    sid = _get_sid(session_id)

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

    speech_ratio = detect_speech_ratio_silero(audio_f32, sr, global_.silero_model, global_.get_speech_timestamps_fn)
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

    feats = None
    try:
        feats = global_.smile.process_signal(audio_f32, sr)
    except Exception:
        with tempfile.NamedTemporaryFile(prefix="opensmile_", suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            sf.write(str(tmp_path), audio_f32, sr)
            feats = global_.smile.process_file(str(tmp_path))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    x = _align_features_with_warning(global_, feats)
    raw = float(global_.model.predict_proba(x)[0, 1])

    prev = _SESSION_EMA.get(sid, None)
    ema = raw if prev is None else (1 - EMA_ALPHA) * prev + EMA_ALPHA * raw
    _SESSION_EMA[sid] = ema

    fatigued = ema >= FATIGUED_THRESHOLD
    weights = three_state_weights(ema)

    return {
        "speaking": True,
        "speech_ratio": float(speech_ratio),
        "fatigue_score_raw": raw,
        "fatigue_score": float(ema),
        "fatigued": bool(fatigued),
        "state_weights": weights,
        "note": "",
    }


# =========================
# API entry: wav_path / wav_url
# =========================
def predict_from_source(
    wav_path: str | None = None,
    wav_url: str | None = None,
    session_id: str | None = None,
    options: dict | None = None,
) -> dict:
    """
    options:
      - resample_to_16k: bool (default True)
      - timeout_sec: float (default 10.0)  # for url
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


# =========================
# FastAPI Server
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _ = get_global_components()
    yield
    # Shutdown (if needed)

app = FastAPI(title="Realtime Silero VAD + Fatigue API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    wav_path: str | None = None
    wav_url: str | None = None
    session_id: str | None = None
    options: dict | None = None


class SessionResetRequest(BaseModel):
    session_id: str | None = None


@app.get("/healthz")
def healthz():
    try:
        _ = get_global_components()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predict_from_source(
            wav_path=req.wav_path,
            wav_url=req.wav_url,
            session_id=req.session_id,
            options=req.options,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_file")
async def predict_file(
    file: UploadFile = File(...),
    session_id: str | None = None,
    resample_to_16k: bool = True,
):
    """
    上传 wav 文件进行预测。
    - file: multipart/form-data file
    - session_id: query param（可选）
    - resample_to_16k: query param（默认 True）
    """
    try:
        content = await file.read()
        bio = io.BytesIO(content)

        audio, sr_in = sf.read(bio, always_2d=True)  # (n, c)
        ch_in = int(audio.shape[1])

        audio_mono = _ensure_mono(_to_float32(audio))

        sr_used = int(sr_in)
        if resample_to_16k:
            audio_mono = _resample(audio_mono, sr_in=sr_in, sr_out=SR)
            sr_used = SR

        core = predict_audio(audio_f32=audio_mono, sr=sr_used, session_id=session_id)

        return {
            "ok": True,
            "session_id": _get_sid(session_id),
            "input": {
                "source_kind": "upload",
                "source_value": file.filename,
                "sr_in": int(sr_in),
                "sr_used": int(sr_used),
                "channels_in": int(ch_in),
                "duration_sec": float(len(audio_mono)) / float(sr_used) if sr_used > 0 else 0.0,
            },
            "output": core,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/reset")
def session_reset(req: SessionResetRequest):
    sid = _get_sid(req.session_id)
    existed = sid in _SESSION_EMA
    _SESSION_EMA.pop(sid, None)
    return {"ok": True, "session_id": sid, "existed": existed}


# =========================
# Direct run support (optional)
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)