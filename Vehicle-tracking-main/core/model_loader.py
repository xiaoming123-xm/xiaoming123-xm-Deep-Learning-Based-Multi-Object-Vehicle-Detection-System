# -*- coding: utf-8 -*-
"""
ModelLoadThread：后台线程负责加载 YOLOv8 模型
使用 Ultralytics 内置 ByteTrack，无需 DeepSORT
"""

import os
import torch

from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class ModelLoadThread(QThread):
    progress     = pyqtSignal(int, str)   # (进度百分比, 描述文字)
    finished_sig = pyqtSignal(bool, str)  # (是否成功, 消息)

    def __init__(self, opt):
        super().__init__()
        self.opt   = opt
        self.model = None
        # ByteTrack 由 model.track() 内置调用，无需预加载额外权重

    def run(self):
        try:
            self.progress.emit(20, "加载 YOLOv8 模型...")

            if os.path.exists(self.opt.yolo_weights):
                self.model = YOLO(self.opt.yolo_weights, verbose=False)
            else:
                self.progress.emit(30, "下载 yolov8n.pt ...")
                self.model = YOLO("yolov8n.pt", verbose=False)
                try:
                    import shutil
                    if os.path.exists("yolov8n.pt"):
                        shutil.copy("yolov8n.pt", self.opt.yolo_weights)
                except Exception:
                    pass

            if self.opt.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')

            # Ultralytics 内置 ByteTrack，model.track() 调用时自动启用
            self.progress.emit(100, "YOLOv8 + ByteTrack 加载完成")
            self.finished_sig.emit(True, "YOLO 加载成功（内置 ByteTrack 跟踪，ID 稳定）")

        except Exception as e:
            self.finished_sig.emit(False, f"模型加载失败: {e}")