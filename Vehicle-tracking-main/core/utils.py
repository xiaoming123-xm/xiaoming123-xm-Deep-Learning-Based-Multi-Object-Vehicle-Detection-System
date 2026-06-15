# -*- coding: utf-8 -*-
"""
工具函数：目录创建、路径计算、颜色转换、视频路径查找
"""

import os
import json
import yaml

from PyQt5.QtGui import QColor

from core.constants import DEFAULT_VIDEO_DIR, VEHICLE_COLORS, DEFAULT_COLOR


def ensure_dir(p):
    """如果目录不存在则创建"""
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def read_yml(path):
    """读取 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_annotation_path(video_path, output_dir):
    """根据视频路径生成对应 JSON 标注文件路径"""
    name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{name}_annotations.json")


def get_rendered_video_path(video_path, output_dir):
    """根据视频路径生成对应已渲染视频路径"""
    name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{name}_out_annotated.mp4")


def load_existing_annotations(video_path, output_dir):
    """尝试加载已有的 JSON 标注文件，失败返回 None"""
    p = get_annotation_path(video_path, output_dir)
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 加载标注失败: {e}")
    return None


def get_video_start_dir():
    """获取文件选择对话框的起始目录（按优先级检测）"""
    candidates = [
        DEFAULT_VIDEO_DIR,
        r"D:\python_project\yolov8_car\Vehicle-tracking-main",
        r"D:\python_project\yolov8_car",
        r"D:\python_project",
        os.path.expanduser("~"),
        os.getcwd(),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def get_color(vtype):
    """根据车辆类型返回 BGR 颜色元组"""
    return VEHICLE_COLORS.get(vtype, DEFAULT_COLOR)


def bgr_to_qcolor(bgr_tuple, alpha=255):
    """BGR 元组转 QColor（PyQt 使用 RGB）"""
    b, g, r = bgr_tuple
    return QColor(r, g, b, alpha)