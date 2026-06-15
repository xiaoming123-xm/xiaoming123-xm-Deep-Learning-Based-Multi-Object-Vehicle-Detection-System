# -*- coding: utf-8 -*-
"""
OPT 配置类：读取 YAML、初始化模型路径、设备选择、输出目录等
"""

import os
import torch
import yaml

from core.utils import ensure_dir, read_yml


class OPT:
    def __init__(self, config_path):
        self.config_path  = os.path.abspath(config_path)
        self.config_dir   = os.path.dirname(self.config_path)
        self.project_root = os.path.dirname(self.config_dir)

        self.model_root = r"D:\python_project\yolov8_car\Vehicle-tracking-main\models"
        ensure_dir(self.model_root)

        try:
            self.config = read_yml(self.config_path)
        except Exception:
            self.config = {}

        # 设备选择
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        # 检测参数
        self.classes    = self.config.get('classes',    [0, 1, 2, 3, 5, 7])
        self.conf_thres = self.config.get('conf_thres', 0.25)
        self.iou_thres  = self.config.get('iou_thres',  0.5)

        # 模型权重（ByteTrack 内置于 Ultralytics，无需额外权重）
        self.yolo_weights = os.path.join(self.model_root, "yolov8n.pt")

        # 输出控制
        self.save_json = True
        self.save_vid  = True
        self.output_dir = (
            r"D:\python_project\yolov8_car\Vehicle-tracking-main"
            r"\application\main\inference\output"
        )
        ensure_dir(self.output_dir)

        # 轨迹参数
        self.max_trail_len   = 10   # 滚动窗口长度（无选中时显示）
        self.trail_thickness = 6    # 追踪轨迹基础粗细（px）
        self.trail_alpha     = 0.85

        self.max_det = 1000
        self.fourcc  = 'mp4v'

    def _abs(self, rel):
        """将相对路径转为绝对路径（相对于项目根目录）"""
        p = os.path.join(self.project_root, rel)
        return os.path.normpath(p) if os.path.exists(p) else os.path.abspath(rel)