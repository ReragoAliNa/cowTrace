# 牛只行为与体征监测系统 (Cow Behavior and Physical Sign Monitoring System)

这是一个基于深度学习和多目标跟踪（MOT）的智能牛只行为与健康体征监测系统。该系统能够对牛群进行高精度的个体识别与追踪，自动分析并分类牛只的日常行为与生理异常（如站立、行走、躺卧、跛行、发情、觅食等），并实时将监测数据同步至云端可视化控制台，助力智慧牧场的科学管理。

---

## 🌟 核心功能

1. **自适应图像预处理 (`CowImagePreprocessor`)**
   - **动态亮度增强**：利用自适应直方图均衡化（CLAHE），估算图像亮度并动态调节对比度限制，显著提升夜间或低光照牧场环境下的牛只画面清晰度。
   - **双边去噪滤波**：应用双边滤波器平滑灰尘与噪点，在消除传感器噪点的同时，完美保留牛只的边缘信息（对后续关键点和分割至关重要）。
   - **等比例缩放与填充**：确保输入模型的图像不发生纵横比形变，支持将推理坐标无损映射回原视频空间。

2. **多任务深度学习引擎 (`LSNetEngine`)**
   - **TensorRT 加速推理**：支持 NVIDIA TensorRT 优化，提供 FP16 低延迟高吞吐量推理。在无 GPU 加速卡的环境中，可自动无缝切换至 PyTorch 或 CPU 模拟模式。
   - **多任务并行输出**：完美承载目标检测（Bounding Box）、实例分割（Segmentation Mask）以及姿态估计（Pose Estimation）等多任务预测。

3. **双版本多目标跟踪器 (`Tracker`)**
   - **Rust 加速版跟踪器 (`cow_trace_rs`)**：使用 Rust 语言配合 `Rayon` 库进行多线程并行 IoU 关联计算，能够高效应对高密集牛群追踪场景。
   - **Python 原生版跟踪器**：基于卡尔曼滤波（Kalman Filter）和匈牙利匹配算法（Hungarian Method），具有极强的跨平台兼容性和稳定性。

4. **体征异常与行为分析器 (`CowBehaviorAnalyzer`)**
   - **几何与运动学启发式分析**：通过融合牛只历史轨迹质心速度、外接框宽高比等多维指标，准确分类基础状态。
   - **基于姿态的跛行检测**：利用姿态估计输出的肩部、脊椎中点和臀部关键点，计算**脊椎弯曲弧度（Back Arch Angle）**，灵敏捕捉牛只因疼痛引起的拱背跛行姿态。
   - **多模型结果融合**：当引入专属行为分类模型 (`behavior.pt`) 时，系统通过 IoU 空间重叠关联自动校准行为事件，提供精准的精细行为分类。

5. **云端监测控制台 (`Cloud System`)**
   - **Go 语言后台**：基于 Go 构建的极简高性能 Web 服务器，支持数据批量上报与轻量级 JSON 数据库持久化。
   - **Web 前端 Dashboard**：提供数据可视化面板，包含牛群状态饼图、警报总览、单牛历史轨迹溯源、视频日志播放与历史行为事件时间轴。

---

## 📂 项目目录结构

```text
cowTrace/
├── main.py                 # 系统总控主入口，运行端到端视频分析流水线
├── config.py               # 预处理参数、异常阈值及云端同步参数配置
├── benchmark_tracker.py    # 追踪器性能基准测试脚本（Python vs Rust）
├── LICENSE                 # 开源许可协议
├── core/                   # 核心算法实现
│   ├── __init__.py
│   ├── preprocessor.py     # CLAHE增强、去噪、等比例缩放与坐标逆映射
│   ├── tracker.py          # 卡尔曼滤波跟踪器（支持Rust/Python动态切换）
│   └── behavior_analyzer.py# 体征行为诊断（躺卧检测、跛行拱背角计算）
├── models/                 # 权重与引擎存放目录
│   ├── __init__.py
│   ├── lsnet_engine.py     # TensorRT & PyTorch 推理封装引擎
│   ├── best.pt             # 默认牛只目标检测与Re-ID模型
│   └── behavior.pt         # 默认细粒度行为分类模型
├── cow_trace_rs/           # Rust 跟踪器加速扩展包
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── src/
│       └── lib.rs          # 基于 PyO3 导出的高性能追踪计算库
├── cloud/                  # Go云端服务与前端控制台
│   ├── main.go             # Go Web服务器入口
│   ├── handlers.go         # API 处理逻辑
│   ├── db.go               # JSON 数据文件存取
│   ├── cloud_server.exe    # Windows平台预编译的云端服务器
│   └── web/
│       └── index.html      # 可视化监控大屏网页
├── utils/
│   └── visualizer.py       # 视频渲染可视化辅助模块（绘制轨迹、信息框）
├── videos/                 # 存放测试视频文件的目录
└── outputs/                # 生成的带追踪标注视频、图片帧和CSV日志的输出目录
```

---

## 🛠️ 环境准备与安装

### 1. Python 环境及依赖

系统运行需要 Python 3.8 或更高版本。请在终端执行以下命令安装必要依赖：

```bash
pip install numpy opencv-python scipy filterpy ultralytics
```

> [!NOTE]
> 如果需要在本地使用 TensorRT 推理加速，需额外安装 `tensorrt` 和 `pycuda` 库，并确保已正确配置 CUDA/cuDNN 环境。

### 2. 高性能 Rust 跟踪器编译（可选）

如需启用 Rust 加速跟踪引擎，需要本地安装有 [Rust 编译器和 Cargo](https://rustup.rs/)。

在 `cowTrace` 项目根目录下运行：

```bash
pip install maturin
cd cow_trace_rs
maturin develop
```

这会将 Rust 代码编译并直接作为 Python 包 `cow_trace_rs` 安装到当前 Python 虚拟环境中。

---

## 🚀 运行指南

### 1. 运行云端 Web 控制台（推荐先启动）

云端控制台负责接收客户端上报的数据并进行大屏展示。

* **直接运行预编译程序** (Windows)：
  双击或在终端运行 `cloud/cloud_server.exe`。
* **从源码编译运行**：
  若已安装 Go 环境，可在 `cloud/` 目录下执行：
  ```bash
  cd cloud
  go run .
  ```

启动成功后，可在浏览器中访问：`http://localhost:8082`。

### 2. 运行视频分析流水线

运行主程序，对视频文件（或系统模拟背景）进行处理：

```bash
python main.py
```

在程序启动时，系统会进入**交互式命令行引导**：
1. **选择跟踪器引擎**：`[1] Rust Tracker (推荐)` 或 `[2] Python Tracker`。
2. **选择处理视频**：系统会扫描 `videos/` 目录下的所有可用视频（若目录为空，将使用内置的合成绿地背景进行模拟运行演示）。
3. **输入处理时长**：输入希望处理的视频秒数限制（例如：`10` 表示只处理前 10 秒；直接回车将处理完整视频）。

**运行产物**：
处理完成后，系统会在 `outputs/<视频名称>/` 目录下生成：
- `output_tracked.mp4`：带标注信息的追踪视频。
- `output_tracked.webm`：适用于浏览器直接播放的 WebM 格式视频。
- `behavior_log.csv`：记录各帧中每只牛的坐标及状态（Status）的日志表格。
- `frame_xxx.png`：关键帧截图。

### 3. 追踪器基准性能测试

系统提供了一个测试脚本，用于对比 Rust 多线程跟踪器与原生 Python 跟踪器在不同牛只规模下的处理速度：

```bash
python benchmark_tracker.py
```

该脚本将输出在目标规模为 5、15、50 和 100 只牛时，两种跟踪器的单次更新耗时（ms）与帧率（Hz），并计算出 Rust 带来的加速倍数。

---

## ⚙️ 系统参数调整

你可以在项目根目录下的 `config.py` 中微调各项核心参数：

| 参数项 | 默认值 | 作用描述 |
| :--- | :--- | :--- |
| `PREPROCESS_CLAHE_CLIP` | `2.0` | CLAHE 对比度限制上限，调大可增强暗部细节但会增加噪声 |
| `LIMPING_THRESHOLD` | `0.5` | 跛行行为诊断灵敏度阈值 |
| `LYING_THRESHOLD` | `0.6` | 躺卧姿态判定灵敏度阈值 |
| `TRACKING_MAX_AGE` | `30` | 允许牛只遮挡或消失的最大帧数，超过该值则销毁追踪ID |
| `CLOUD_SYNC_ENABLED` | `True` | 是否开启实时同步数据至 Go 云端后台 |
| `CLOUD_SYNC_BATCH_SIZE`| `25` | 缓存上报的批次大小，增大可降低网络开销 |
| `MAX_PROCESSING_FRAMES`| `None` | 非交互模式下处理的最大帧数限制 |

---

## 📜 许可证

本项目基于 [MIT License](LICENSE) 协议开源。
