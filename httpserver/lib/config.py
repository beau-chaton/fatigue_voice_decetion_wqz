"""
lib/config.py
统一配置加载器

从 httpserver/config.json 读取所有运行时参数，
以扁平属性方式提供给各模块使用。

用法：
    from lib.config import cfg
    sr = cfg.sr
"""

import json
from pathlib import Path


class _Config:
    """单例配置对象，启动时从 config.json 加载一次。"""

    _loaded = False

    def __init__(self):
        self.host: str = "127.0.0.1"
        self.port: int = 8000
        self.sr: int = 16000
        self.silero_repo_dir: Path = Path("third_party/silero-vad")
        self.silero_jit_path: Path = Path("assets/silero_vad.jit")
        self.min_speech_ratio: float = 0.12
        self.model_path: Path = Path("sleepy_opensmile_logreg.joblib")
        self.ema_alpha: float = 0.5
        self.freeze_when_no_speech: bool = True
        self.fatigued_threshold: float = 0.7
        self.missing_feature_warn_ratio: float = 0.05
        self.energetic_threshold: float = 0.4
        self.fatigue_threshold: float = 0.7
        self.additive: float = 0.2
        self.debug: bool = False

    def load(self, path: str | Path | None = None) -> None:
        """
        从 JSON 文件加载配置，覆盖默认值。

        Args:
            path: config.json 路径，默认为 httpserver/config.json
        """
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # server
        server = data.get("server", {})
        self.host = str(server.get("host", self.host))
        self.port = int(server.get("port", self.port))

        # audio
        audio = data.get("audio", {})
        self.sr = int(audio.get("sr", self.sr))

        # vad
        vad = data.get("vad", {})
        self.silero_repo_dir = Path(vad.get("silero_repo_dir", str(self.silero_repo_dir)))
        self.silero_jit_path = Path(vad.get("silero_jit_path", str(self.silero_jit_path)))
        self.min_speech_ratio = float(vad.get("min_speech_ratio", self.min_speech_ratio))

        # model
        model = data.get("model", {})
        self.model_path = Path(model.get("model_path", str(self.model_path)))

        # predict
        predict = data.get("predict", {})
        self.ema_alpha = float(predict.get("ema_alpha", self.ema_alpha))
        self.freeze_when_no_speech = bool(predict.get("freeze_when_no_speech", self.freeze_when_no_speech))
        self.fatigued_threshold = float(predict.get("fatigued_threshold", self.fatigued_threshold))
        self.missing_feature_warn_ratio = float(predict.get("missing_feature_warn_ratio", self.missing_feature_warn_ratio))

        # scoring
        scoring = data.get("scoring", {})
        self.energetic_threshold = float(scoring.get("energetic_threshold", self.energetic_threshold))
        self.fatigue_threshold = float(scoring.get("fatigue_threshold", self.fatigue_threshold))
        self.additive = float(scoring.get("additive", self.additive))

        # debug
        self.debug = bool(data.get("debug", self.debug))

        self._loaded = True


cfg = _Config()
cfg.load()
