# Deep Learning Based Multi-Object Vehicle Detection and Tracking System

## Project Description

**Project Title**: Deep Learning Based Multi-Object Vehicle Detection and Tracking System

This project is my **undergraduate graduation design**. It is developed based on open-source projects, with extensive research, code analysis, refactoring, and feature improvements carried out independently.

### Key Improvements & Innovations (Compared to Original)

- Upgraded from **YOLOv5 + DeepSORT** to **YOLOv8 + ByteTrack** (more stable tracking and cleaner code)
- Developed a complete **PyQt5 graphical user interface** with video playback and control features
- Added **scrollable "Detected Vehicles" list** on the right panel, supporting **multi-vehicle tracking**
- Implemented **filter rendering**: When vehicles are selected, only tracked vehicles are displayed
- Added **progressive thickened trajectory** with gradient thickness (thinner at tail, thicker at head)
- Designed a **three-level priority playback system**: JSON Annotation > Rendered Video > New Detection
- Supports saving detection results as **JSON annotations** and **annotated MP4 videos**
- Fully modular code structure (`core / tracker / ui`) for better maintainability

---

### System Features

- Real-time vehicle detection and tracking
- Click on vehicle boxes or list items to toggle multi-tracking
- Video progress bar, pause, resume, and loop playback
- Real-time system logging
- Supports both **webcam** and **local video files**
- JSON-based annotation system for fast replay and filtering

---

### Tech Stack

- **Detection & Tracking**: Ultralytics YOLOv8 + ByteTrack
- **GUI Framework**: PyQt5
- **Language**: Python 3.8+
- **Libraries**: OpenCV, NumPy, Torch, PyYAML

---

# Recommended Python version
Python 3.8 - 3.10

# Install dependencies
pip install -r requirements.txt

# 基于深度学习的多目标车辆检测与追踪系统

## 项目说明

**项目名称**：基于深度学习的多目标车辆检测与追踪系统（Vehicle Tracking System）

本项目是本人的**本科毕业设计**，在原开源项目的基础上，**自主搜集资料、分析代码、进行大量修改和功能升级**后完成。

**核心改进与创新点**（相对原版本）：
- 从 YOLOv5 + DeepSORT 升级为 **YOLOv8 + ByteTrack**（追踪更稳定，代码更简洁高效）
- 开发了完整的 **PyQt5 图形用户界面**，支持实时视频播放、进度控制等
- 新增**右侧已检测车辆滚动列表**，支持**多选追踪**（可同时追踪多辆车）
- 实现**过滤渲染**：选中车辆后，其他车辆标注自动隐藏，仅显示追踪目标
- 支持**渐进式加粗轨迹**，视觉效果更清晰
- 采用 **JSON 标注 + 已渲染视频 + 全新检测** 三级优先级播放机制
- 支持保存标注信息（JSON）和带标注视频（MP4）
- 优化了模型加载、轨迹管理、ID映射等模块，代码结构更加清晰模块化

---

### 项目演示

（在此处插入你最终的演示 GIF 或视频截图）

**系统界面主要功能**：
- 视频实时检测与追踪
- 点击车辆或列表项进行多选追踪
- 进度条跳转、暂停、循环播放
- 系统日志实时显示
- 支持摄像头与本地视频文件输入

---

### 技术栈

- **检测与追踪**：Ultralytics YOLOv8 + ByteTrack
- **界面框架**：PyQt5
- **编程语言**：Python 3.8+
- **其他**：OpenCV, NumPy, Torch, PyYAML

---

### 运行环境

```bash
# 推荐 Python 版本
Python 3.8 - 3.10

# 安装依赖
pip install -r requirements.txt
