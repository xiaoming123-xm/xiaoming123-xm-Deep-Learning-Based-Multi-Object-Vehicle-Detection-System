# -*- coding: utf-8 -*-
"""
detect_mode.py
实时 AI 检测模式（ByteTrack）：逐帧推理、轨迹更新、标注写入、视频保存
该模块被 TrackingThread._detect_mode() 调用
"""

import cv2
import time
import torch

from core.utils import (
    get_color,
    get_rendered_video_path,
    load_existing_annotations,
    ensure_dir,
)


def draw_box(im0, bbox, gid, vtype, conf=None, highlighted=False):
    """
    在帧上绘制检测框和标签。
    highlighted=True 时额外绘制白色外框（追踪目标）。
    """
    color = get_color(vtype)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    th = 3 if highlighted else 2

    if highlighted:
        cv2.rectangle(im0, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)

    cv2.rectangle(im0, (x1, y1), (x2, y2), color, th)

    lbl = f"{vtype}-{gid}" + (f" {conf:.2f}" if conf is not None else "")
    (tw, th_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(im0, (x1, y1 - th_t - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(im0, lbl, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return im0


def run_detect_loop(thread):
    """
    ByteTrack 检测主循环。
    thread 是 TrackingThread 实例，通过它访问：
      thread.model, thread.cap, thread.opt,
      thread.trail_mgr, thread.selected_ids,
      thread.all_seen_vehicles, thread.annotations,
      thread.global_id_counter, thread.id_map,
      thread.is_running, thread.is_paused,
      thread._seek_req, thread.current_frame_idx,
      thread.total_frames, thread.prev_veh
    以及 emit 信号：
      thread.status_msg, thread.frame_changed,
      thread.frame_ready, thread.model_loading
    """
    opt   = thread.opt
    model = thread.model

    fps    = thread.cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thread.total_frames = int(thread.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thread.status_msg.emit(
        f"视频: {width}×{height} @ {fps:.1f}fps  总帧: {thread.total_frames}"
    )

    # 初始化标注结构
    thread.annotations = {
        'video_path':          str(thread.video_path),
        'frame_rate':          fps,
        'resolution':          (width, height),
        'frames':              [],
        'selected_global_ids': [],
    }

    # 初始化带标注视频写入器
    annotated_writer    = None
    annotated_out_path  = None
    if opt.save_vid and isinstance(thread.video_path, str):
        ensure_dir(opt.output_dir)
        out_path = get_rendered_video_path(thread.video_path, opt.output_dir)
        annotated_writer   = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*opt.fourcc),
            fps, (width, height)
        )
        annotated_out_path = out_path
        thread.status_msg.emit(f"将保存带标注视频: {out_path}")

    delay_ms   = max(1, int(1000 / fps))
    frame_idx  = 0
    start_time = time.time()

    while thread.is_running and thread.cap.isOpened():

        # 暂停等待
        while thread.is_paused and thread.is_running:
            thread.msleep(40)

        # seek 请求处理
        if thread._seek_req >= 0:
            tgt = thread._seek_req
            thread._seek_req = -1
            thread.cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
            frame_idx = tgt
            thread.trail_mgr.clear()
            thread.prev_veh = {}

        ret, frame = thread.cap.read()
        if not ret:
            break

        frame_idx += 1
        thread.current_frame_idx = frame_idx
        thread.frame_changed.emit(frame_idx, thread.total_frames)

        im0              = frame.copy()
        frame_for_writer = frame.copy() if annotated_writer else None
        cur_veh          = {}
        frame_vehicles   = []

        # ── ByteTrack 推理 ──────────────────────────────────────────
        try:
            results = model.track(
                frame,
                conf=opt.conf_thres,
                iou=opt.iou_thres,
                classes=opt.classes,
                max_det=opt.max_det,
                persist=True,              # 跨帧保持同一轨迹 ID
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except Exception as e:
            thread.status_msg.emit(f"推理错误: {e}")
            results = []

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes     = r.boxes.xyxy.cpu().numpy()
                confs     = r.boxes.conf.cpu().numpy()
                clss      = r.boxes.cls.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    bbox   = boxes[i]
                    conf_v = float(confs[i])
                    cid    = int(clss[i])
                    lid    = int(track_ids[i])
                    vtype  = model.names.get(cid, 'unknown')

                    # ByteTrack ID → 全局 ID
                    gid = thread._get_gid(lid)

                    thread.trail_mgr.update(gid, bbox)
                    if gid not in thread.all_seen_vehicles:
                        thread.all_seen_vehicles[gid] = vtype

                    is_sel = gid in thread.selected_ids if thread.selected_ids else False
                    color  = get_color(vtype)

                    # 全量标注写入渲染视频（不受追踪过滤影响）
                    if frame_for_writer is not None:
                        trail_w = thread.trail_mgr.trails[gid]
                        frame_for_writer = thread.trail_mgr.draw(frame_for_writer, trail_w, color)
                        frame_for_writer = draw_box(frame_for_writer, bbox, gid, vtype, conf_v, False)

                    # 过滤渲染到实时画面
                    if not thread.selected_ids or is_sel:
                        trail = thread.trail_mgr.get_trail(gid, is_sel)
                        im0   = thread.trail_mgr.draw(im0, trail, color)
                        im0   = draw_box(im0, bbox, gid, vtype, conf_v, is_sel)

                    vi = {
                        'global_id':  gid,
                        'local_id':   lid,
                        'type':       vtype,
                        'bbox':       [float(b) for b in bbox],
                        'confidence': conf_v,
                        'trail':      thread.trail_mgr.to_json_list(gid),
                        'selected':   is_sel,
                    }
                    frame_vehicles.append(vi)
                    cur_veh[gid] = vi

        thread.prev_veh = cur_veh
        thread.annotations['frames'].append({
            'frame_idx': frame_idx,
            'vehicles':  frame_vehicles,
        })

        # 写入渲染视频
        if frame_for_writer is not None and annotated_writer and annotated_writer.isOpened():
            elapsed_d = time.time() - start_time
            fps_d     = frame_idx / (elapsed_d + 1e-6)
            cv2.putText(frame_for_writer,
                        f"FPS:{fps_d:.1f}  Veh:{len(frame_vehicles)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            annotated_writer.write(frame_for_writer)

        # OSD 叠加 FPS 信息
        elapsed  = time.time() - start_time
        proc_fps = frame_idx / (elapsed + 1e-6)
        cv2.putText(im0, f"FPS:{proc_fps:.1f}  Veh:{len(frame_vehicles)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 发送帧信号
        qt_img = thread._to_qimage(im0)
        h, w   = im0.shape[:2]
        thread.frame_ready.emit(qt_img, frame_idx, {
            'frame_idx':         frame_idx,
            'num_vehicles':      len(frame_vehicles),
            'vehicles':          frame_vehicles,
            'total_vehicles':    thread.global_id_counter - 1,
            'fps':               proc_fps,
            'elapsed':           elapsed,
            'video_size':        (w, h),
            'all_seen_vehicles': dict(thread.all_seen_vehicles),
        })

        if not thread.is_paused:
            thread.msleep(delay_ms)

    # ── 收尾 ────────────────────────────────────────────────────────
    if annotated_writer and annotated_writer.isOpened():
        annotated_writer.release()
        if annotated_out_path and annotated_out_path:
            thread.status_msg.emit(
                f"✓ 带标注视频已保存: {annotated_out_path}"
            )

    return frame_idx