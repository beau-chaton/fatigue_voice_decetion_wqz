# 疲劳语音检测项目（openSMILE + 逻辑回归）

本项目通过语音特征（eGeMAPS）训练一个二分类模型，输出语音片段的疲劳概率（`P(Sleepy)`），并提供实时检测脚本。

## 1. 项目结构

- `feature_create.py`：批量读取 `voice_data/openslr` 下的音频，统一预处理为 16kHz 单声道 3 秒，并提取 eGeMAPS 特征到 CSV。
- `train_sleepy_score.py`：按说话人留一测试训练模型，并保存模型。
- `train_sleepy_score_notest.py`：使用全部数据训练模型，并保存模型。
- `realtime_fatigue_vad.py`：实时录音 + VAD（是否有人说话）+ 疲劳分数输出。
- `features_egemaps.csv`：提取后的训练特征文件。
- `sleepy_opensmile_logreg.joblib`：训练得到的模型文件。

## 2. 环境安装

建议 Python 3.10-3.12（Windows）。

```bash
pip install -r requirements.txt
```

说明：
- 在 Windows 下，`webrtcvad` 可能需要本地编译环境；本项目已使用 `webrtcvad-wheels`，安装后可直接 `import webrtcvad`。

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

确保已有模型文件 `sleepy_opensmile_logreg.joblib`，然后运行：

```bash
python realtime_fatigue_vad.py
```

脚本行为：
- 每 `3` 秒录一段语音（可在脚本里调 `WINDOW_SECONDS` / `HOP_SECONDS`）
- 使用 WebRTC VAD 判断窗口内是否有人说话
- 若有人说话，输出疲劳分数 `fatigue_score`（0 到 1）
- 当前默认阈值：`fatigue_score >= 0.7` 判为疲劳

## 7. 常见问题

1. `No wav found under ...`
- 检查 `voice_data/openslr` 路径是否存在且包含 `.wav` 文件。

2. `ModuleNotFoundError: webrtcvad`
- 重新执行 `pip install -r requirements.txt`，确认 `webrtcvad-wheels` 安装成功。

3. 首次运行特征提取较慢
- `librosa/scipy` 首次加载、Windows 防病毒扫描都可能导致启动慢，通常后续会更快。

## 8. 快速执行顺序

```bash
pip install -r requirements.txt
python feature_create.py
python train_sleepy_score.py
python realtime_fatigue_vad.py
```
