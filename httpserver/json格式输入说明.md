必须满足的条件
wav_path 和 wav_url 二选一至少提供一个
代码逻辑：wav_path 优先；只有没给 wav_path 才会用 wav_url
两个都不填会报错：Either wav_path or wav_url must be provided
可选字段
session_id：可选
options：可选
resample_to_16k：可选（默认 true）
timeout_sec：可选（默认 10.0，仅 wav_url 用）