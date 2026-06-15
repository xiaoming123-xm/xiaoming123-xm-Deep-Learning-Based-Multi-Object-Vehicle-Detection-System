# -*- coding: utf-8 -*-
"""
video_label.py
ClickableVideoLabel：可响应鼠标点击的视频显示控件
点击视频上的检测框区域时，发出 vehicle_clicked(gid) 信号
"""

from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QMouseEvent


class ClickableVideoLabel(QLabel):
    vehicle_clicked = pyqtSignal(int)   # 发出被点击车辆的 global_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("border: 2px solid #2e3248; background-color: #0d0f18;")
        self.setText("等待视频启动...")
        self.setFont(QFont("Microsoft YaHei", 14))

        self._vehicles = []
        self._vid_size = (0, 0)
        self._pix_rect = None   # (ox, oy, dw, dh) 实际渲染区域

    def setPixmap(self, pix):
        super().setPixmap(pix)
        lw, lh = self.width(),  self.height()
        pw, ph = pix.width(),   pix.height()
        if pw == 0 or ph == 0:
            self._pix_rect = None
            return
        scale = min(lw / pw, lh / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        ox, oy = (lw - dw) // 2, (lh - dh) // 2
        self._pix_rect = (ox, oy, dw, dh)

    def set_vehicles(self, vehicles: list, vid_size: tuple):
        """每帧更新当前帧的车辆列表和视频原始尺寸"""
        self._vehicles = vehicles
        self._vid_size = vid_size

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton and self._vehicles and self._pix_rect:
            ox, oy, dw, dh = self._pix_rect
            cx, cy = ev.x() - ox, ev.y() - oy
            if 0 <= cx <= dw and 0 <= cy <= dh:
                vw, vh = self._vid_size
                if vw and vh:
                    vx = cx * vw / dw
                    vy = cy * vh / dh
                    for v in self._vehicles:
                        b = v['bbox']
                        if b[0] <= vx <= b[2] and b[1] <= vy <= b[3]:
                            self.vehicle_clicked.emit(v['global_id'])
                            return
        super().mousePressEvent(ev)