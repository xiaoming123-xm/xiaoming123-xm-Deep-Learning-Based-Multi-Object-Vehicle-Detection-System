# -*- coding: utf-8 -*-
"""
tracking_thread.py
TrackingThread：后台视频处理主线程
负责三种播放模式的调度：
  模式0 — 播放已渲染 MP4（无过滤）
  模式1 — JSON 标注重建画面（支持多选过滤 + 渐进轨迹）
  模式2 — 实时 ByteTrack AI 检测（保存标注 + 视频）
"""

import os
import cv2
import time
from collections import defaultdict

import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from core.utils import (
    get_color,
    get_rendered_video_path,
    load_existing_annotations,
    ensure_dir,
)
from core.model_loader import ModelLoadThread
from tracker.trail_manager import TrailManager
from tracker.annotation_io import preload_seen_from_anno, save_annotations
from tracker.detect_mode import run_detect_loop, draw_box


class TrackingThread(QThread):
    frame_ready    = pyqtSignal(QImage, int, dict)
    truly_finished = pyqtSignal()
    status_msg     = pyqtSignal(str)
    frame_changed  = pyqtSignal(int, int)
    model_loading  = pyqtSignal(int, str)

    def __init__(self, video_path, opt, load_existing=True, force_redetect=False):
        super().__init__()
        self.video_path     = video_path
        self.opt            = opt
        self.load_existing  = load_existing
        self.force_redetect = force_redetect

        self.is_running = True
        self.is_paused  = False
        self.is_loop    = True
        self._seek_req  = -1

        self.cap          = None
        self.video_writer = None
        self.model_thread = None
        self.model        = None

        # ID 映射
        self.global_id_counter = 1
        self.id_map            = defaultdict(dict)
        self.prev_veh          = {}

        # 轨迹管理器
        self.trail_mgr = TrailManager(
            max_len   = opt.max_trail_len,
            thickness = opt.trail_thickness,
            alpha     = opt.trail_alpha,
        )

        self.annotations       = None
        self.current_frame_idx = 0
        self.total_frames      = 0

        # 多选追踪
        self.selected_ids      = set()
        self.all_seen_vehicles = {}   # gid -> vtype

    # ================================================================
    # 控制接口
    # ================================================================
    def pause(self):              self.is_paused = True
    def resume(self):             self.is_paused = False
    def set_loop(self, v):        self.is_loop = v
    def set_selected_ids(self, ids: set): self.selected_ids = set(ids)

    def toggle_selected_id(self, gid: int):
        if gid in self.selected_ids:
            self.selected_ids.discard(gid)
        else:
            self.selected_ids.add(gid)

    def seek(self, n):
        if self.total_frames > 0:
            n = max(0, min(n, self.total_frames - 1))
        self._seek_req = n

    def stop(self):
        self.is_running = False
        self.is_paused  = False
        if self.model_thread and self.model_thread.isRunning():
            self.model_thread.terminate()
        self.msleep(80)
        self._free()

    def _free(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
        except Exception:
            pass

    # ================================================================
    # ID 工具
    # ================================================================
    def _get_gid(self, local_id):
        """ByteTrack 本地 ID → 项目全局 ID"""
        key = os.path.basename(str(self.video_path))
        if local_id not in self.id_map[key]:
            self.id_map[key][local_id] = self.global_id_counter
            self.global_id_counter += 1
        return self.id_map[key][local_id]

    # ================================================================
    # 帧转换
    # ================================================================
    def _to_qimage(self, im0):
        rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    # ================================================================
    # 主调度（优先顺序：JSON > 渲染视频 > AI 检测）
    # ================================================================
    def run(self):
        try:
            if (self.load_existing
                    and not self.force_redetect
                    and isinstance(self.video_path, str)):

                # ── 优先级1：JSON 标注 ──────────────────────────────
                anno = load_existing_annotations(self.video_path, self.opt.output_dir)
                if anno and anno.get('frames'):
                    self.status_msg.emit("✓ 找到已有标注，JSON 播放（支持车辆过滤）...")
                    self.annotations = anno
                    preload_seen_from_anno(anno, self.all_seen_vehicles)
                    self._play_anno()
                    return

                # ── 优先级2：已渲染视频 ─────────────────────────────
                rendered_path = get_rendered_video_path(self.video_path, self.opt.output_dir)
                if anno:
                    json_rendered = anno.get('rendered_video_path')
                    if json_rendered and os.path.exists(json_rendered):
                        rendered_path = json_rendered
                if os.path.exists(rendered_path):
                    self.status_msg.emit(
                        f"✓ 已渲染视频直接播放（无 JSON，不支持过滤）: "
                        f"{os.path.basename(rendered_path)}"
                    )
                    self._play_rendered_video(rendered_path, anno)
                    return

            # ── 优先级3：全新 AI 检测 ───────────────────────────────
            self.status_msg.emit("启动模型检测...")
            self._detect_mode()

        except Exception as e:
            self.status_msg.emit(f"线程异常: {e}")
        finally:
            self.truly_finished.emit()

    # ================================================================
    # 模式0：播放已渲染 MP4
    # ================================================================
    def _play_rendered_video(self, rendered_path, anno=None):
        total_vehicles = 0
        anno_frames    = []

        if anno:
            total_vehicles = anno.get('total_vehicles', 0)
            anno_frames    = anno.get('frames', [])
            preload_seen_from_anno(anno, self.all_seen_vehicles)

        cap = cv2.VideoCapture(rendered_path)
        if not cap.isOpened():
            self.status_msg.emit(f"无法打开渲染视频: {rendered_path}")
            if anno and anno.get('frames'):
                self.annotations = anno
                self._play_anno()
            return

        fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.status_msg.emit(
            f"渲染视频: {width}×{height} @ {fps:.1f}fps  总帧: {self.total_frames}"
        )

        delay_ms   = max(1, int(1000 / fps))
        frame_idx  = 0
        start_time = time.time()
        anno_len   = len(anno_frames)

        while self.is_running:
            while self.is_paused and self.is_running:
                self.msleep(40)

            if self._seek_req >= 0:
                tgt = self._seek_req
                self._seek_req = -1
                cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                frame_idx = tgt

            ret, frame = cap.read()
            if not ret:
                if self.is_loop and self.is_running:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx  = 0
                    start_time = time.time()
                    continue
                break

            frame_idx += 1
            self.current_frame_idx = frame_idx
            self.frame_changed.emit(frame_idx, self.total_frames)

            frame_vehicles = []
            if anno_len > 0 and (frame_idx - 1) < anno_len:
                raw_vehs = anno_frames[frame_idx - 1].get('vehicles', [])
                for v in raw_vehs:
                    v_copy = v.copy()
                    v_copy['selected'] = (
                        (v.get('global_id') in self.selected_ids)
                        if self.selected_ids else False
                    )
                    frame_vehicles.append(v_copy)

            elapsed  = time.time() - start_time
            proc_fps = frame_idx / (elapsed + 1e-6)

            qt_img = self._to_qimage(frame)
            h, w   = frame.shape[:2]
            self.frame_ready.emit(qt_img, frame_idx, {
                'frame_idx':         frame_idx,
                'num_vehicles':      len(frame_vehicles),
                'vehicles':          frame_vehicles,
                'total_vehicles':    total_vehicles,
                'fps':               proc_fps,
                'elapsed':           elapsed,
                'video_size':        (w, h),
                'all_seen_vehicles': dict(self.all_seen_vehicles),
            })

            if not self.is_paused:
                self.msleep(delay_ms)

        cap.release()
        self.status_msg.emit("播放结束")

    # ================================================================
    # 模式1：JSON 标注重建画面
    # ================================================================
    def _play_anno(self):
        max_id = 0
        for frm in self.annotations.get('frames', []):
            for v in frm.get('vehicles', []):
                max_id = max(max_id, v.get('global_id', 0))
        self.global_id_counter = max_id + 1

        anno_frames = self.annotations.get('frames', [])
        anno_len    = len(anno_frames)

        cap_src  = self.video_path if isinstance(self.video_path, str) else int(self.video_path)
        self.cap = cv2.VideoCapture(cap_src)
        if not self.cap.isOpened():
            self.status_msg.emit(f"无法打开视频: {self.video_path}")
            return

        fps           = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        width         = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height        = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.status_msg.emit(
            f"视频: {width}×{height} @ {fps:.1f}fps  总帧: {self.total_frames}"
        )

        delay_ms   = max(1, int(1000 / fps))
        frame_idx  = 0
        start_time = time.time()

        while self.is_running:
            while self.is_paused and self.is_running:
                self.msleep(40)

            if self._seek_req >= 0:
                tgt = self._seek_req
                self._seek_req = -1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                frame_idx = tgt
                # 跳转时清空渐进轨迹，避免轨迹残留
                self.trail_mgr.clear()

            ret, frame = self.cap.read()
            if not ret:
                if self.is_loop and self.is_running:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx  = 0
                    start_time = time.time()
                    self.trail_mgr.clear_full()  # 循环时重置全程轨迹
                    continue
                break

            frame_idx += 1
            self.current_frame_idx = frame_idx
            self.frame_changed.emit(frame_idx, self.total_frames)

            im0            = frame.copy()
            frame_vehicles = []

            if anno_len > 0 and (frame_idx - 1) < anno_len:
                anno = anno_frames[frame_idx - 1]

                for v in anno.get('vehicles', []):
                    bbox  = v['bbox']
                    gid   = v['global_id']
                    vtype = v['type']
                    conf  = v.get('confidence')

                    # 注册已见车辆
                    if gid not in self.all_seen_vehicles:
                        self.all_seen_vehicles[gid] = vtype

                    # 从 JSON 恢复滚动窗口轨迹
                    if v.get('trail'):
                        self.trail_mgr.restore_from_json(gid, v['trail'])

                    # 累积全程轨迹（渐进显示）
                    self.trail_mgr.update_full_from_bbox(gid, bbox)

                    is_selected = (gid in self.selected_ids) if self.selected_ids else False
                    should_draw = (not self.selected_ids) or is_selected

                    if should_draw:
                        color = get_color(vtype)
                        trail = self.trail_mgr.get_trail(gid, is_selected)
                        im0   = self.trail_mgr.draw(im0, trail, color)
                        im0   = draw_box(im0, bbox, gid, vtype, conf, is_selected)

                    v_copy = v.copy()
                    v_copy['selected'] = is_selected
                    frame_vehicles.append(v_copy)

            elapsed  = time.time() - start_time
            proc_fps = frame_idx / (elapsed + 1e-6)
            cv2.putText(im0, f"FPS:{proc_fps:.1f}  Veh:{len(frame_vehicles)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            qt_img = self._to_qimage(im0)
            h, w   = im0.shape[:2]
            self.frame_ready.emit(qt_img, frame_idx, {
                'frame_idx':         frame_idx,
                'num_vehicles':      len(frame_vehicles),
                'vehicles':          frame_vehicles,
                'total_vehicles':    self.annotations.get('total_vehicles', max_id),
                'fps':               proc_fps,
                'elapsed':           elapsed,
                'video_size':        (w, h),
                'all_seen_vehicles': dict(self.all_seen_vehicles),
            })

            if not self.is_paused:
                self.msleep(delay_ms)

        self._free()
        self.status_msg.emit("播放结束")

    # ================================================================
    # 模式2：实时 AI 检测（委托给 detect_mode.py）
    # ================================================================
    def _detect_mode(self):
        # 加载模型
        self.model_thread = ModelLoadThread(self.opt)
        self.model_thread.progress.connect(self.model_loading)
        self.model_thread.finished_sig.connect(self._on_model_loaded)
        self.model_thread.start()

        import time as _time
        t0 = _time.time()
        while self.model_thread.isRunning() and self.is_running:
            self.msleep(150)
            if _time.time() - t0 > 120:
                self.status_msg.emit("模型加载超时")
                self.model_thread.terminate()
                return

        if not self.is_running or not self.model:
            return

        # 打开视频
        cap_src  = self.video_path if isinstance(self.video_path, str) else int(self.video_path)
        self.cap = cv2.VideoCapture(cap_src)
        if not self.cap.isOpened():
            self.status_msg.emit(f"无法打开视频: {self.video_path}")
            return

        # 执行检测循环（在 detect_mode.py 中实现）
        frame_idx = run_detect_loop(self)

        self._free()

        # 保存 JSON 标注
        self.annotations['selected_global_ids'] = list(self.selected_ids)
        ok, msg = save_annotations(
            self.annotations,
            str(self.video_path),
            self.opt.output_dir,
            self.global_id_counter,
            self.selected_ids,
            self.all_seen_vehicles,
        )
        self.status_msg.emit(msg)
        self.status_msg.emit(
            f"检测完成！帧:{frame_idx}  目标:{self.global_id_counter - 1}\n"
            f"下次播放将加载 JSON 标注，支持多选过滤。"
        )

        # 检测完成后切换到 JSON 回放
        if self.is_loop and self.is_running and isinstance(self.video_path, str):
            anno = load_existing_annotations(self.video_path, self.opt.output_dir)
            if anno and anno.get('frames'):
                self.status_msg.emit("切换到 JSON 标注回放模式（支持车辆过滤）...")
                self.annotations = anno
                preload_seen_from_anno(anno, self.all_seen_vehicles)
                self.trail_mgr.clear_full()
                self._play_anno()

    def _on_model_loaded(self, ok, msg):
        self.status_msg.emit(msg)
        if ok:
            self.model = self.model_thread.model