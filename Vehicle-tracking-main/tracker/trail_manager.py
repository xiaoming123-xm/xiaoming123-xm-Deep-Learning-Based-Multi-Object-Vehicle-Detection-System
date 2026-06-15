# -*- coding: utf-8 -*-
"""
trail_manager.py
轨迹点的更新（滚动窗口 + 全程累积）和渐进加粗绘制
"""

import cv2
from collections import defaultdict


class TrailManager:
    """
    管理每辆车的两种轨迹：
      trails      — 滚动窗口（最近 max_len 帧），无追踪时显示
      full_trails — 全程累积，追踪选中时显示（渐进加粗）
    """

    def __init__(self, max_len: int = 10, thickness: int = 6, alpha: float = 0.85):
        self.max_len   = max_len
        self.thickness = thickness
        self.alpha     = alpha

        self.trails      = defaultdict(list)  # gid -> [(cx, cy), ...]
        self.full_trails = defaultdict(list)  # gid -> [(cx, cy), ...]

    def clear(self):
        """清空所有轨迹（跳帧 / 循环时调用）"""
        self.trails.clear()
        self.full_trails.clear()

    def clear_full(self):
        """只清空全程累积轨迹"""
        self.full_trails.clear()

    def update(self, gid: int, bbox):
        """
        根据检测框计算车辆底部中心点，更新两种轨迹。
        bbox: [x1, y1, x2, y2]
        """
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int(bbox[1] + (bbox[3] - bbox[1]) * 0.75)

        # 全程累积
        self.full_trails[gid].append((cx, cy))

        # 滚动窗口
        self.trails[gid].append((cx, cy))
        if len(self.trails[gid]) > self.max_len:
            self.trails[gid].pop(0)

    def restore_from_json(self, gid: int, trail_list: list):
        """从 JSON 标注恢复滚动窗口轨迹"""
        self.trails[gid] = [tuple(p) for p in trail_list]

    def update_full_from_bbox(self, gid: int, bbox):
        """仅更新全程累积轨迹（JSON 回放模式专用）"""
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int(bbox[1] + (bbox[3] - bbox[1]) * 0.75)
        last = self.full_trails[gid][-1] if self.full_trails[gid] else None
        if last != (cx, cy):
            self.full_trails[gid].append((cx, cy))

    def get_trail(self, gid: int, is_selected: bool) -> list:
        """返回用于绘制的轨迹点列表"""
        return self.full_trails[gid] if is_selected else self.trails[gid]

    def to_json_list(self, gid: int) -> list:
        """将滚动窗口轨迹序列化为 JSON 可存储的列表"""
        return [list(p) for p in self.trails[gid]]

    def draw(self, frame, pts: list, color: tuple):
        """
        渐进加粗绘制轨迹：
        线段从尾部（细）到头部（粗），体现行驶方向。
        """
        if len(pts) < 2:
            return frame

        ov   = frame.copy()
        n    = len(pts)
        base = self.thickness

        for i in range(1, n):
            t  = i / n                                   # 0→1：尾→头
            th = max(2, int(base * (0.3 + 0.7 * t)))    # 2px → base px
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]),     int(pts[i][1]))
            cv2.line(ov, p1, p2, color, th, cv2.LINE_AA)

        cv2.addWeighted(ov, self.alpha, frame, 1 - self.alpha, 0, frame)
        return frame