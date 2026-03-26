"""
lib/model.py
全局模型组件的单例加载与管理

包含：
  - GlobalComponents  : 持有所有运行时组件的数据类
  - get_global_components : 懒加载单例，进程内只初始化一次

依赖：
  - lib.config  : 模型文件路径
  - lib.vad     : Silero VAD 加载
"""

from dataclasses import dataclass, field
from typing import Any

import joblib
import opensmile

from lib.config import cfg
from lib.vad import build_silero_vad_local


@dataclass
class GlobalComponents:
    """进程级单例，持有推理所需的所有重型对象。"""

    model: Any
    feature_cols: list[str]
    smile: opensmile.Smile
    silero_model: Any
    get_speech_timestamps_fn: Any
    warned_missing_features: bool = field(default=False)


_GLOBAL: GlobalComponents | None = None


def get_global_components() -> GlobalComponents:
    """
    懒加载并缓存全局组件（进程内只执行一次）。

    加载顺序：
      1. joblib 加载 sklearn 模型 + feature_cols
      2. 初始化 opensmile.Smile（eGeMAPSv02 Functionals）
      3. 本地加载 Silero VAD 模型

    Returns:
        GlobalComponents 单例
    """
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL

    bundle = joblib.load(cfg.model_path)
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
