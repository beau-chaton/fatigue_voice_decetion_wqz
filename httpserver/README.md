# httpserver — 疲劳检测 FastAPI HTTP 服务

本目录是以 `realtime_silero_vad_fastapi.py` 为主文件的**独立可运行服务包**，
包含主文件运行所需的所有依赖文件副本，可直接部署。

---

## 目录结构

```
httpserver/
├── realtime_silero_vad_fastapi.py   # 主服务文件（FastAPI HTTP Server）
├── sleepy_opensmile_logreg.joblib   # 疲劳分类模型（LogReg + eGeMAPS 特征）
├── requirements.txt                  # Python 依赖列表
├── json格式输入说明.md               # API 请求格式说明
├── README.md                         # 本文件
├── assets/
│   └── silero_vad.jit               # Silero VAD TorchScript 模型
└── third_party/
    └── silero-vad/                  # Silero VAD 本地源码（离线加载）
        └── src/
            └── silero_vad/
                └── utils_vad.py     # VAD 核心函数（get_speech_timestamps）
```

---

## 路径依赖说明

主文件 `realtime_silero_vad_fastapi.py` 中使用相对路径加载以下资源：

| 配置变量 | 路径 | 说明 |
|---|---|---|
| `MODEL_PATH` | `sleepy_opensmile_logreg.joblib` | scikit-learn 分类模型 |
| `SILERO_REPO_DIR` | `third_party/silero-vad` | Silero VAD 本地仓库 |
| `SILERO_JIT_PATH` | `assets/silero_vad.jit` | Silero VAD JIT 模型 |

> ⚠️ **必须从 `httpserver/` 目录下启动服务**，否则相对路径无法正确解析。

---

## Python 版本要求

| 依赖库 | 版本 | 最低 Python |
|---|---|---|
| numpy | 2.4.3 | ≥ 3.11 |
| pandas | 3.0.1 | ≥ 3.11 |
| scikit-learn | 1.8.0 | ≥ 3.11 |
| torch | ≥ 2.0.0 | ≥ 3.11 |

**推荐 Python 版本：3.11 / 3.12 / 3.13**（官方测试覆盖范围）

> ⚠️ Python 3.10 及以下版本**不支持**本项目的依赖库，请勿使用。

---

## 快速启动

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# 方式一：直接运行
cd httpserver
python realtime_silero_vad_fastapi.py

# 方式二：使用 uvicorn（推荐生产环境）
cd httpserver
uvicorn realtime_silero_vad_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/healthz

---

## API 端点

| Method | Path | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查，确认模型已加载 |
| POST | `/predict` | 通过文件路径或 URL 预测疲劳度 |
| POST | `/predict_file` | 上传 WAV 文件预测疲劳度 |
| POST | `/session/reset` | 重置指定 session 的 EMA 状态 |

详见 `json格式输入说明.md` 或 `/docs` 接口文档。
