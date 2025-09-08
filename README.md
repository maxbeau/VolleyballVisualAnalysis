# 🏐 Volleyball Visual Analysis (Roboflow + ByteTrack)

本项目基于 Roboflow 云推理与 Supervision ByteTrack，实现：
- 单图推理并从场地分割轮廓提取四个角点
- 视频多模型检测合并 + ByteTrack 多目标追踪 + 可视化导出
- 基于球运动的回合切割与简易前端查看

---

## ✨ 功能
- 🏟️ 场地角点提取：从分割轮廓拟合四角点（TL, TR, BR, BL），并输出 JSON
- 🎯 多目标追踪：合并多模型检测结果，使用 ByteTrack 输出 ID、轨迹与可视化视频
- 🕹️ 回合切割：球速/缺失驱动的简单状态机（pipeline）
- 📊 简易前端：Streamlit 预览视频与回合列表

---

## 📦 环境准备
1) Python 3.10+，并准备虚拟环境：
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

pip install inference-sdk supervision opencv-python ffmpeg-python streamlit numpy
```

---

## 🔑 配置与路径
在项目根目录创建并编辑 `.env`：
```env
ROBOFLOW_API_KEY=YOUR_API_KEY
ROBOFLOW_MODEL_ID=volleyball_v2/3
ROBOFLOW_CONFIDENCE=0.25

# 可选：本地或托管视频
VIDEO_PATH=data/input.mov
VIDEO_URL=

# 采样与回合切割参数
INFER_FPS=10
START_SPEED_PPS=120
START_MIN_CONSEC=3
END_ABSENCE_SEC=1.5
END_SLOW_PPS=60
END_MIN_CONSEC=6
MIN_SEGMENT_SEC=1.8
```

主要文件与输出路径：
- 单图推理与场地角点：
  - 输入：`data/input.png`
  - 输出图：`outputs/input_annotated.png`
  - 角点 JSON：`outputs/input_annotated_court_corners.json`
- 视频追踪（ByteTrack）：
  - 输入：`data/input.mov`（或自定义）
  - 叠加视频：`outputs/track_overlay.mp4`
  - 轨迹 JSON：`outputs/track_overlay.json`
- 回合切割：
  - 结果 JSON：`outputs/segments.json`

---

## ▶️ 使用与实现路径（当前精简版）

1) 视频抽帧检测（仅球模型）：`detect_ball.py`
```bash
python3 detect_ball.py
```
说明：
- 读取 `.env` 的 `ROBOFLOW_API_KEY`，使用模型 `volleyball_v2/3`。
- 按 `INFER_FPS` 抽帧调用云推理，逐帧缓存 JPEG+JSON 到 `outputs/preds`。
- 合并输出 `outputs/ball_detections.jsonl`（每行一个帧的预测）。

2) 原始帧完整叠加 + 线性插值：`overlay_ball_full.py`
```bash
python3 overlay_ball_full.py
```
说明：
- 逐帧读取原视频（保持原始帧率与时长）。
- 在两次采样帧之间对球框做线性插值；间隔过大或开头/结尾空窗时，启用“持有最近框（hold）”作为回退。
- 输出 `outputs/ball_overlay_full.mp4`。

---

## 📁 关键文件（当前精简版）
- `detect_ball.py`：抽帧调用 Roboflow 球模型并缓存预测 + JSONL 汇总
- `overlay_ball_full.py`：原始帧视频叠加，线性插值 + 可配置持有回退
- `utils.py`：公共工具（读取 .env、保证目录存在、选择视频路径）
- `roboflow_client.py`：Roboflow API 调用封装（独立模块，便于后续替换为其它提供方/本地模型）
- `detect_court.py`：低频采样（默认每 5 秒）检测球场轮廓，按帧缓存 + 综合多次结果（中值 + EMA）生成精确四角点

---

## ⚠️ 注意事项
- 云推理成本：多模型 × 抽帧率 → 调用次数；建议先低 FPS 验证再放开
- 绿屏/播放异常：优先 `--codec avc1`，或使用 VLC 播放，或退回 MJPG/AVI
- 分割多边形：若未返回多边形则回退到矩形框；四角点顺序统一为 TL,TR,BR,BL

---

## ⚙️ 可配置项（.env）
- `ROBOFLOW_API_KEY`: Roboflow API Key（必填）
- `VIDEO_PATH`: 视频路径（默认自动找 `data/input.mov`/`data/input.mp4`）
- `ROBOFLOW_CONFIDENCE`: 云推理置信度（默认 0.25）
- `INFER_FPS`: 抽帧频率（默认 5）
- `MAX_FRAMES`: 抽帧上限（测试用，可省略）
- `CACHE_DIR`: 帧与预测缓存目录（默认 `outputs/preds`）
- `COMBINED_JSONL`: 汇总预测文件（默认 `outputs/ball_detections.jsonl`）
- `BALL_OVERLAY_FULL`: 完整带框视频输出（默认 `outputs/ball_overlay_full.mp4`）
- `BALL_CLASSES`: 球类别筛选（默认 `ball,volleyball`）
- `MAX_INTERP_GAP_FRAMES`: 允许插值的最大帧距（默认 60）
- `HOLD_MODE`: 回退持有策略 `prev|next|both|none`（默认 `prev`）
- `HOLD_TTL_FRAMES`: 回退“持有最近框”的最大帧数（默认 30）
- `OVERLAY_MIN_CONF`: 叠加时的最小置信度过滤（默认 0）
- Court 检测：
  - `COURT_INTERVAL_SEC`: 采样间隔秒（默认 5）
  - `COURT_INTERVAL_SEC_MIN`: 动态加密采样的最小间隔（默认 2）
  - `COURT_CHANGE_THRESH_PX`: 触发加密采样的像素阈值（默认 20）
  - `COURT_CACHE_DIR`: 球场检测缓存目录（默认 `outputs/court_preds`）
  - `COURT_COMBINED_JSONL`: 球场检测汇总（默认 `outputs/court_detections.jsonl`）
  - `COURT_INTEGRATED_JSON`: 综合四角点输出（默认 `outputs/court_corners_integrated.json`）

---

## 🚀 建议的下一步
- 透视矫正：用四角点做单应性，将轨迹投影到标准场地坐标系，便于统计与可视化热区
- 追踪优化：为不同类别设置独立 ByteTrack 参数；对球采用高更新率、对球员更稳定
- 预测缓存：将每帧云推理结果缓存到本地，避免重复调用，支持断点续跑
- 团队/号码识别：接入号码 OCR 或分色聚类，绑定球员身份，输出每人触球/跑动距离等
- UI 强化：在 Streamlit 中叠加轨迹、射门落点、回合过滤条件等交互
