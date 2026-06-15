# -*- coding: utf-8 -*-
"""
annotation_io.py
JSON 标注文件的读取、预加载、保存逻辑
"""

import json
import os
from datetime import datetime

from core.utils import (
    get_annotation_path,
    get_rendered_video_path,
    ensure_dir,
)


def preload_seen_from_anno(anno, all_seen_vehicles: dict):
    """
    从 JSON 标注中预加载所有已见车辆到 all_seen_vehicles。
    优先使用 anno 中保存好的 all_seen_vehicles 字段；
    否则遍历 frames 重建。
    """
    saved = anno.get('all_seen_vehicles')
    if saved and isinstance(saved, dict):
        for k, v in saved.items():
            try:
                all_seen_vehicles[int(k)] = v
            except Exception:
                pass
        return

    for frm in anno.get('frames', []):
        for v in frm.get('vehicles', []):
            gid   = v.get('global_id')
            vtype = v.get('type', 'unknown')
            if gid is not None and gid not in all_seen_vehicles:
                all_seen_vehicles[gid] = vtype


def save_annotations(
    annotations: dict,
    video_path: str,
    output_dir: str,
    global_id_counter: int,
    selected_ids: set,
    all_seen_vehicles: dict,
):
    """
    将 annotations 字典序列化写入 JSON 文件。
    同时更新 total_frames / total_vehicles / annotation_time 等元数据。
    """
    try:
        p = get_annotation_path(video_path, output_dir)
        ensure_dir(os.path.dirname(p))
        rendered_path = get_rendered_video_path(video_path, output_dir)

        annotations.update({
            'total_frames':        len(annotations.get('frames', [])),
            'total_vehicles':      global_id_counter - 1,
            'annotation_time':     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'selected_global_ids': list(selected_ids),
            'all_seen_vehicles':   all_seen_vehicles,
            'rendered_video_path': rendered_path if os.path.exists(rendered_path) else None,
        })

        with open(p, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        return True, f"✓ 标注已保存: {p}"
    except Exception as e:
        return False, f"保存标注失败: {e}"