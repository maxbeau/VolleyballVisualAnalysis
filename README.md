# 🏐 Volleyball Visual Analysis

基于 Roboflow 云推理的排球视频分析工具集：球/球场检测、时序平滑（Kalman+RTS）、可选“重力外推”、检测后处理与视频叠加输出。

---

## ✨ 功能
- 检测缓存：按固定 FPS 抽帧调用 Roboflow，将结果保存为逐帧 JSON 与汇总 JSONL
- 球场角点：低频采集 + Kalman/RTS 时序平滑与插值，输出逐帧角点
- 时序平滑：前向 Kalman + 反向 RTS 插值、门控与去噪，遮挡段更稳定
- 重力外推：在预测步注入竖直加速度，使轨迹外推更接近抛物线
- 软权重过滤：按长宽比对可疑框做置信度衰减，减少误检干扰
- 叠加可视化：在原视频绘制球框与可选球场边界，输出新视频
  - 球场方向自适应：四朝向模板打分 + EMA 平滑 + 连胜门槛 + 切换锁定
  - 中线与三米线：按标准 18×9m 模型投影中线（x=9m）与三米线（x=6m、12m）
 

---

## 📦 环境准备
- Python 3.9+（推荐 3.10+）
- 安装依赖：
```bash
# 创建并激活虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 使用 requirements.txt 安装所有依赖
pip install -r requirements.txt
```
说明：`requirements.txt` 包含 `inference-sdk`, `opencv-python`, `numpy`, `python-dotenv` 等所有必需的库。

---

## 🔑 配置与路径
- 在根目录创建/编辑 `.env` 文件以配置项目（如 `ROBOFLOW_API_KEY`）。
- 所有可调参数均通过 `.env` 文件管理，由 `core/config.py` 统一加载。
- **视频路径**：如果 `.env` 中未设置 `VIDEO_PATH`，程序会自动查找并使用 `data/input.mov` 或 `data/input.mp4`。
- 关键路径默认值：
  - 检测缓存：`outputs/preds/`（JPEG+JSON）
  - 检测汇总：`outputs/ball_detections.jsonl`
  - 叠加视频：`outputs/ball_overlay_full.mp4`
  - 球场跟踪：`outputs/court_tracking.jsonl`
  
核心配置分组：
- 输入与缓存：`VIDEO_PATH`，`INFER_FPS`，`CACHE_DIR`，`COMBINED_JSONL`
- 叠加输出：`BALL_OVERLAY_FULL`，`OVERLAY_MIN_CONF`，`SHOW_BOX_LABELS`，`OVERLAY_CODEC`
- 平滑与插值：`MAX_INTERP_GAP_FRAMES`，`HOLD_MODE`，`HOLD_TTL_FRAMES`
- 门控与重力：`OBS_GATE_CHISQ_THRESH`，`OBS_GATE_USE_CONF`，`GRAVITY_PPS2`
- 软权重过滤：`FILTER_MIN_ASPECT_RATIO`，`FILTER_MAX_ASPECT_RATIO`，`FILTER_AR_SOFT_ALPHA`
- 运动学过滤（可选）：`KINEMATIC_FILTER_ENABLE`，`KIN_MAX_SPEED_PX_PER_S`，`KIN_MAX_ACCEL_PX_PER_S2`，`KIN_MAX_DIR_CHANGE_DEG`，`KIN_MAX_SIZE_FRAC_PER_S`
- 球场叠加：`COURT_OVERLAY`，`COURT_OVERLAY_METHOD(=timeseries)`，`COURT_COLOR`，`COURT_THICKNESS`
  - 线条颜色扩展：`COURT_CENTER_COLOR`（中线，默认 0,255,255），`COURT_ATTACK_COLOR`（三米线，默认 255,0,255）

---

## ▶️ 快速开始

1) 视频抽帧检测（仅球模型，内存推理）：
```bash
python3 scripts/run_ball_detect.py
```
说明：
- 读取 `.env` 的 `ROBOFLOW_API_KEY`，使用模型 `volleyball_v2/3`。
- 按 `INFER_FPS` 抽帧调用云推理，默认使用“内存推理”（不落盘中间 JPEG/JSON）。
- 如需保留缓存，可设置：`BALL_SAVE_JPEGS=true`、`BALL_SAVE_FRAME_JSON=true`。
- 合并输出 `outputs/ball_detections.jsonl`（每行一个帧的预测）。

2) 原始帧完整叠加 + 卡尔曼/RTS 平滑：
```bash
python3 scripts/run_overlay.py
```
说明：
- 逐帧读取原视频（保持原始帧率与时长）。
- 使用 Kalman + RTS 对检测做插值和平滑；相邻观测间隔过大或首尾空窗时，启用“持有最近框（hold）”回退。
- 支持“长宽比软权重”“运动学过滤（可选）”与“重力外推”提升遮挡段的外推质量。
- 输出 `outputs/ball_overlay_full.mp4`。

### 运动学过滤（可选）
启用后，将在平滑前按图像空间速度/加速度/转向/尺寸变化做硬门控，排除明显不可能为排球的帧。

`.env` 示例参数：

```
KINEMATIC_FILTER_ENABLE=true
# 建议根据视频分辨率与 FPS 微调
KIN_MAX_SPEED_PX_PER_S=2200       # 最大像素速度
KIN_MAX_ACCEL_PX_PER_S2=18000     # 最大像素加速度
KIN_MAX_DIR_CHANGE_DEG=135        # 相邻速度向量允许的最大转角
KIN_MAX_SIZE_FRAC_PER_S=4.0       # 每秒尺寸相对变化上限（宽/高分别检查）
```

说明：
- 超过像素速度/加速度上限的帧直接丢弃；
- 相邻速度向量转角过大（例如>135°）视为不连续，丢弃；
- 尺寸变化速率过大（透视缩放不应突变）视为假阳性，丢弃；
- 过滤在 Kalman+RTS 平滑之前生效，避免错误观测干扰轨迹。

3) 场地“采集-处理”解耦与动态跟踪（推荐）：

- 采集（仅云推理与缓存，不做融合）：
```bash
python3 scripts/run_court_detect.py
```
- 处理（Kalman+RTS 平滑与逐帧插值）：
```bash
python3 scripts/run_court_processing.py --detections-jsonl outputs/court_detections.jsonl --tracking-jsonl outputs/court_tracking.jsonl
```
说明：
- 采集阶段专注于按时间间隔抽帧并保存“原始预测+角点”，形成可复用数据资产（JSONL+缓存帧）。
- 处理阶段对角点做“时间序列平滑与插值”，输出逐帧角点 `outputs/court_tracking.jsonl`，用于真正的“动态跟踪”。
- 叠加使用 `COURT_OVERLAY_METHOD=timeseries`，按帧读取 `COURT_TRACKING_JSONL` 绘制动态场地。
- 采集阶段采用内存推理（避免磁盘 I/O）；如需缓存采样帧 JPEG，可在 `.env` 设 `COURT_SAVE_JPEGS=true`。

---

## 📁 关键文件
- `ball/detect.py`：Roboflow 抽帧检测、缓存与 JSONL 汇总
- `visualization/overlay.py`：叠加渲染（Kalman+RTS、软权重、重力、可选场地）
- `analysis/smoothing.py`：Kalman + RTS 平滑、观测门控与重力注入
- `court/utils.py`：球场几何工具（角点解析与排序）
- `court/detect.py`：仅采集球场检测结果（抽样帧 JPEG+原始 JSON、合并 JSONL）
- `court/config.py`：Court 跟踪/门控/模板评分/Kalman 等参数集中管理
- `court/tracker.py`：LK+RANSAC 单应 + 模板评分 + 几何门控 + Kalman 平滑
- `court/io.py`：关键帧 JSONL 加载与稳健去极值
- `court/processing.py`：编排器（读取关键帧→跟踪→写 `court_tracking.jsonl`）
- `court/smoothing.py`：2D 常速度 Kalman（世界坐标轻量去噪，用于轨迹）
- `core/roboflow_client.py`：Roboflow SDK 轻封装与网络设置
- `core/utils.py`：读取 `.env`、保证目录存在、选择视频路径
- `court/homography.py`：基于时序角点（tracking）计算单应性并生成鸟瞰图
- `analysis/trajectory.py`：融合球轨迹与单应性，计算世界坐标与物理量
 

---

## ⚠️ 注意事项
- 云推理成本：抽帧率越高、视频越长 → 成本越高；建议先低 FPS 验证
- 播放兼容：优先 `avc1`，异常时用 VLC 或切换 `MJPG`/`AVI`
- 坐标系：像素 y 轴向下为正；重力取正值（向下）
 

---

## ⚙️ 可配置项（节选）
- 输入与缓存：`VIDEO_PATH`，`INFER_FPS`，`CACHE_DIR`，`COMBINED_JSONL`
- 叠加输出：`BALL_OVERLAY_FULL`，`OVERLAY_MIN_CONF`，`SHOW_BOX_LABELS`，`OVERLAY_CODEC`
- 平滑与插值：`MAX_INTERP_GAP_FRAMES`，`HOLD_MODE`，`HOLD_TTL_FRAMES`
- 门控与重力：`OBS_GATE_CHISQ_THRESH`，`OBS_GATE_USE_CONF`，`GRAVITY_PPS2`
- 软权重过滤：`FILTER_MIN_ASPECT_RATIO`，`FILTER_MAX_ASPECT_RATIO`，`FILTER_AR_SOFT_ALPHA`
- 球场叠加：`COURT_OVERLAY`，`COURT_OVERLAY_METHOD(=timeseries)`，`COURT_COLOR`，`COURT_THICKNESS`

---

## 🚀 调参与建议
- 重力：`GRAVITY_PPS2 ≈ 9.8 * px_per_m`；未知比率可先试 150–250
- 长宽比：先宽后紧，区间外用 `FILTER_AR_SOFT_ALPHA` 控制衰减强度
- 门控：`OBS_GATE_CHISQ_THRESH≈18.4`（4 维观测约 3σ），低置信度时更严格
- 插值：`MAX_INTERP_GAP_FRAMES` 控制遮挡段插值长度，过大可能漂移

如需更多定制（ROI 过滤、单应透视、标准场地坐标系等），可在此基础上扩展。

---

## 🗺️ 单应性与鸟瞰图（推荐且唯一方案）

仅使用“时序平滑后的角点”（`outputs/court_tracking.jsonl`）来估计稳定的单应性矩阵，并生成球场鸟瞰图：

```bash
python3 scripts/run_court_homography.py \
  --tracking-jsonl outputs/court_tracking.jsonl \
  --output-h-npy outputs/court_homography.npy \
  --output-h-meta outputs/court_homography.json \
  --birdseye-jpg outputs/court_birdseye.jpg \
  --scale-px-per-meter 100  # 可选；默认 100px/m → 1800x900
```

说明：
- 读取 `outputs/court_tracking.jsonl` 每帧角点，逐点取中位数，得到鲁棒的 TL,TR,BR,BL 再拟合 H。
- 将 18m×9m 映射到 `scale_px_per_meter` 对应的画布（默认 1800×900）。
- 保存单应性：`outputs/court_homography.npy`（3x3）与元数据 JSON。
- 从原视频取一帧（默认中间帧）透视展开，得到 `outputs/court_birdseye.jpg`。
- 如需固定像素尺寸，可用 `--model-size 1800x900` 覆盖比例参数。

---

## 🧩 球场方向识别与线条渲染（叠加阶段）

- 四朝向模板打分：在模型平面（18×9m）生成模板（外框+中线+两条三米线），对 0/90/180/270° 四种朝向计算模板命中率（投影后命中边缘的比例），择优。
- 平滑与滞后：首帧直接选最优；后续对分数做 EMA 平滑，仅当候选朝向相对当前朝向优势>5% 且连续≥3 帧时才切换；切换后锁定 10 帧避免来回抖动。
- 线条渲染：使用最终朝向对应的单应性投影中线（x=9m）与三米线（x=6m、12m），并绘制外框。

---

## 🧭 轨迹到标准球场坐标系（连续与物理量）

前置：已完成球场时序角点与单应性估计（上一节）。

将平滑后的球中心映射到标准球场（鸟瞰）坐标系，并输出连续轨迹与速度等物理量：

```bash
python3 scripts/run_trajectory_analysis.py
```

输出：
- `outputs/trajectory_world.jsonl`：逐帧世界坐标（像素与米）、速度、累计路程
- `outputs/trajectory_world.csv`：同上 CSV 版本，便于表格/绘图分析
- `outputs/trajectory_birdseye.jpg`：在鸟瞰图上绘制的轨迹折线（绿色起点、红色终点）

要点：
- 先在图像坐标系内用 Kalman+RTS 融合（含可选重力外推/观测门控/hold 回退），再用单应性投影到球场平面；
- 可选在球场坐标内再次做 2D 时序平滑（常速度模型），使轨迹更连续；
- 米制换算依据 `court_homography.json` 的 `scale_px_per_meter`，默认 100px/m（18m×9m→1800×900）。

---

## 🛠️ 球场追踪增强与诊断（本次更新）

- 追踪职责划分：
  - LK/单应负责逐帧估计 `H_prev_curr` 并累积；
  - Kalman 仅在关键帧融合 API 观测，帧间只做预测（不使用光流角点作为测量）。
- 模板工具统一：`court/utils.py` 提供模板构建与打分；tracker 与 overlay 共用，降低分叉。
- 自适应 ROI：基于上一帧位移中位数自适应放大/缩小 LK ROI，提速且稳健。
- 尺度门控：限制每帧尺度变化（默认 ±8%），异常帧拒绝写入。
- 自适应测量噪声：关键帧融合 Kalman 时按模板精度自适应测量噪声 R（高质量→更小 R）。
- 诊断输出：`outputs/court_tracking.jsonl` 每帧写 `info` 指标，`COURT_SHOW_DIAG=true` 可在视频左上角显示关键指标（内点率、重投影误差、条件数、尺度、ROI 比例、模板精度、匹配数等）。

可调环境变量（新增）：
- `LK_ROI_EXPAND_RATIO`（默认 0.12）
- `MAX_SCALE_CHANGE_PER_FRAME`（默认 0.08）
- `KF_ADAPTIVE_FROM_TEMPLATE`（默认 true）
- `KF_R_API_MIN`（默认 0.8）
- `KF_R_API_MAX`（默认 2.5）
- `COURT_SHOW_DIAG`（默认 false）

以上均已在 `core/config.py` 读取，`court/processing.py` 注入 tracker。
