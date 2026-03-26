"""
main.py
疲劳检测 FastAPI HTTP 服务入口

职责：仅保留 FastAPI 应用定义、路由与启动入口。
业务逻辑全部由 lib/ 下各模块提供：
  - lib.audio_io   : 音频加载与格式转换
  - lib.vad        : Silero VAD 语音检测
  - lib.model      : 全局模型组件单例
  - lib.scoring    : 疲劳评分与状态权重
  - lib.predict    : 核心预测逻辑与 session 管理

启动方式（必须在 httpserver/ 目录下运行）：
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
from contextlib import asynccontextmanager

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from lib.audio_io import SR, _ensure_mono, _resample, _to_float32
from lib.model import get_global_components
from lib.predict import (
    _get_sid,
    predict_audio,
    predict_from_source,
    session_ema_reset,
)


# ─── 应用生命周期 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载所有模型组件，避免首次请求延迟。"""
    _ = get_global_components()
    yield


app = FastAPI(
    title="Realtime Silero VAD + Fatigue API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── 请求/响应模型 ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    wav_path: str | None = None
    wav_url: str | None = None
    session_id: str | None = None
    options: dict | None = None


class SessionResetRequest(BaseModel):
    session_id: str | None = None


# ─── 路由 ──────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    """健康检查：确认模型已成功加载。"""
    try:
        _ = get_global_components()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """
    通过本地文件路径或远程 URL 预测疲劳度。

    Body (JSON):
        wav_path   : 本地文件路径（与 wav_url 二选一）
        wav_url    : 远程 WAV URL（与 wav_path 二选一）
        session_id : 会话标识（可选，用于跨帧 EMA）
        options    : { resample_to_16k: bool, timeout_sec: float }
    """
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
    上传 WAV 文件进行疲劳度预测。

    Form params:
        file            : multipart/form-data WAV 文件
        session_id      : 会话标识（query param，可选）
        resample_to_16k : 是否重采样到 16kHz（query param，默认 True）
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
                "duration_sec": (
                    float(len(audio_mono)) / float(sr_used) if sr_used > 0 else 0.0
                ),
            },
            "output": core,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/reset")
def session_reset(req: SessionResetRequest):
    """重置指定 session 的 EMA 疲劳分数状态。"""
    sid, existed = session_ema_reset(req.session_id)
    return {"ok": True, "session_id": sid, "existed": existed}


# ─── 直接运行支持 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
