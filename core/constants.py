# -*- coding: utf-8 -*-
"""
常量定义：车辆颜色映射、默认颜色、默认视频目录
"""

DEFAULT_VIDEO_DIR = r"D:\python_project\yolov8_car\Vehicle-tracking-main\videos"

# BGR 格式颜色（OpenCV 使用 BGR）
VEHICLE_COLORS = {
    'car':        (0,   220,  0),
    'motorcycle': (220,  0,  220),
    'bus':        (220, 220,  0),
    'truck':      (220, 140,  0),
    'person':     (0,   180, 220),
    'bicycle':    (0,   180, 180),
}
DEFAULT_COLOR = (220, 60, 60)