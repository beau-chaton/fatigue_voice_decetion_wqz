# 疲劳语音检测项目（openSMILE + 逻辑回归）

本项目通过语音特征（eGeMAPS）训练一个二分类模型，输出语音片段的疲劳概率（`P(Sleepy)`），并提供两个不同 VAD 后端的实时检测脚本。

## 1. 项目结构

- `feature_create.py`：批量读取 `voice_data/openslr` 下的音频，统一预处理为 16kHz 单声道 3 秒，并提取 eGeMAPS 特征到 CSV。
- `train_sleepy_score.py`：按说话人留一测试训练模型，并保存模型。
- `train_sleepy_score_notest.py`：使用全部数据训练模型，并保存模型。
- `realtime_fatigue_vad.py`：实时录音 + WebRTC VAD（是否有人说话）+ 疲劳分数输出。
- `realtime_fatigue_silero_vad.py`：实时录音 + **Silero VAD**（本地离线，精度更高）+ 疲劳分数输出，支持 EMA 平滑与无语音冻结。
- `features_egemaps.csv`：提取后的训练特征文件。
- `sleepy_opensmile_logreg.joblib`：训练得到的模型文件。
- `assets/silero_vad.jit`：Silero VAD 本地模型权重（TorchScript 格式）。
- `third_party/silero-vad/`：Silero VAD 工具函数源码（离线依赖）。

## 2. 环境安装

建议 Python 3.10-3.12（Windows）。

```bash
pip install -r requirements.txt
```

如需运行 `realtime_fatigue_silero_vad.py`，还需额外安装 PyTorch：

```bash
pip install torch torchaudio
```

说明：
- 在 Windows 下，`webrtcvad` 可能需要本地编译环境；本项目已使用 `webrtcvad-wheels`，安装后可直接 `import webrtcvad`。
- Silero VAD 版本完全离线运行，无需网络，模型文件已包含在 `assets/silero_vad.jit`。

## 3. 数据组织

脚本默认从 `voice_data/openslr` 递归读取 `*.wav`，并依赖目录名自动打标签：

- 目录名包含 `Sleepy` -> `label_sleepy = 1`
- 其他目录 -> `label_sleepy = 0`

示例目录：

```text
voice_data/openslr/
  bea_Amused/
  bea_Neutral/
  bea_Sleepy/
  ...
```

## 4. 生成特征

```bash
python feature_create.py
```

运行后将会：
- 在 `tmp_preproc_3s_16k_mono/` 写入统一采样率和时长的临时 wav
- 生成 `features_egemaps.csv`

## 5. 训练模型

### 5.1 留一说话人测试（推荐先用）

在 `train_sleepy_score.py` 中可修改：
- `TEST_SPEAKER = "bea"`（可改为 `jenie` / `josh` / `sam` 等）

运行：

```bash
python train_sleepy_score.py
```

输出：
- 控制台打印 AUC、AP、分类报告
- 保存 `sleepy_opensmile_logreg.joblib`

### 5.2 全量数据训练（不留测试）

```bash
python train_sleepy_score_notest.py
```

输出：
- 使用全部样本训练后保存 `sleepy_opensmile_logreg.joblib`

## 6. 实时疲劳检测

确保已有模型文件 `sleepy_opensmile_logreg.joblib`，然后选择以下任意一个脚本运行：

### 6.1 WebRTC VAD 版本（轻量，依赖少）

```bash
python realtime_fatigue_vad.py
```

脚本行为：
- 每 `3` 秒录一段语音（可在脚本里调 `WINDOW_SECONDS` / `HOP_SECONDS`）
- 使用 WebRTC VAD 判断窗口内是否有人说话
- 若有人说话，输出疲劳分数 `fatigue_score`（0 到 1）
- 当前默认阈值：`fatigue_score >= 0.7` 判为疲劳

### 6.2 Silero VAD 版本（推荐，精度更高）

```bash
python realtime_fatigue_silero_vad.py
```

脚本行为：
- 每 `3` 秒录一段语音（可在脚本里调 `WINDOW_SECONDS` / `HOP_SECONDS`）
- 使用本地 Silero VAD（`assets/silero_vad.jit`）计算 `speech_ratio`
- `speech_ratio < MIN_SPEECH_RATIO`（默认 0.12）时认为"无人说话"，可选冻结上一次分数
- 若有人说话，提取 eGeMAPS 特征并输出 EMA 平滑后的疲劳分数
- 当前默认阈值：`fatigue_score >= 0.7` 判为疲劳

关键可调参数（脚本顶部 `Config` 区域）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `WINDOW_SECONDS` | `3.0` | 每次录音窗口长度（秒） |
| `HOP_SECONDS` | `3.0` | 输出刷新间隔（秒），设为 `1.0` 可实现重叠滑窗 |
| `MIN_SPEECH_RATIO` | `0.12` | Silero VAD 判"有人说话"的最低语音占比 |
| `EMA_ALPHA` | `0.5` | 指数移动平均平滑系数（0~1，越大越跟随当前值） |
| `FREEZE_WHEN_NO_SPEECH` | `True` | 无语音时是否保持上一次分数而非输出 `None` |
| `FATIGUED_THRESHOLD` | `0.7` | 疲劳判断阈值 |

## 7. 两种 VAD 对比

| 特性 | WebRTC VAD | Silero VAD |
|---|---|---|
| 额外依赖 | `webrtcvad-wheels` | `torch`、`torchaudio` |
| 离线支持 | 是 | 是（需预置 `.jit` 文件） |
| 精度 | 一般 | 较高 |
| 推理速度 | 极快 | 快（CPU 可用） |
| 输出平滑 | 无 | EMA 平滑 |

## 8. 常见问题

1. `No wav found under ...`
   - 检查 `voice_data/openslr` 路径是否存在且包含 `.wav` 文件。

2. `ModuleNotFoundError: webrtcvad`
   - 重新执行 `pip install -r requirements.txt`，确认 `webrtcvad-wheels` 安装成功。

3. `FileNotFoundError: Missing assets/silero_vad.jit`
   - 确认 `assets/silero_vad.jit` 文件存在，该文件为 Silero VAD 离线模型权重。

4. `ImportError: Failed to import get_speech_timestamps`
   - 确认 `third_party/silero-vad/src/silero_vad/utils_vad.py` 文件存在。

5. 首次运行特征提取较慢
   - `librosa/scipy` 首次加载、Windows 防病毒扫描都可能导致启动慢，通常后续会更快。

## 9. 快速执行顺序

```bash
# 安装依赖
pip install -r requirements.txt
pip install torch torchaudio   # Silero VAD 版本需要

# 提取特征
python feature_create.py

# 训练模型
python train_sleepy_score.py

# 实时检测（选一）
python realtime_fatigue_vad.py           # WebRTC VAD 版本
python realtime_fatigue_silero_vad.py   # Silero VAD 版本（推荐）
```
