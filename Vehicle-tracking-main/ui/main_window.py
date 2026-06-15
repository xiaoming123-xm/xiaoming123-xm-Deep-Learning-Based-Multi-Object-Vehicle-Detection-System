# -*- coding: utf-8 -*-
"""
main_window.py
MainWindow：系统主界面
负责视频显示、控制按钮、车辆追踪列表、系统日志、进度条等 UI 逻辑
"""

import os
import yaml

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox,
    QGroupBox, QGridLayout, QTextEdit, QFileDialog, QSplitter,
    QSlider, QCheckBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QFont, QPixmap, QColor, QBrush
from core.config import OPT
from core.constants import VEHICLE_COLORS, DEFAULT_COLOR
from core.utils import (
    ensure_dir,
    get_video_start_dir,
    get_rendered_video_path,
    load_existing_annotations,
)
from tracker.tracking_thread import TrackingThread
from ui.video_label import ClickableVideoLabel
from ui.styles import MAIN_STYLE


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的多目标车辆检测系统 v5.0")

        screen = QApplication.primaryScreen().availableGeometry()
        sw = min(int(screen.width()  * 0.60), 1200)
        sh = min(int(screen.height() * 0.60), 750)
        self.resize(sw, sh)
        self.setMinimumSize(800, 560)
        self.move(
            screen.x() + (screen.width()  - sw) // 2,
            screen.y() + (screen.height() - sh) // 2,
        )

        self.opt          = self._load_config()
        self.track_thread = None
        self.cur_fi       = {}
        self.total_frames = 0
        self.is_playing   = False
        self._drag        = False

        self.tracked_ids       = set()          # 当前追踪的 gid 集合
        self.all_seen_vehicles = {}             # gid -> vtype
        self.veh_list_items    = {}             # gid -> QListWidgetItem
        self._current_video_path = None

        self.setStyleSheet(MAIN_STYLE)
        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick_elapsed)
        self.timer.start(300)

        self.log("✓ 系统启动 v5.0")
        self.log(f"  运行设备: {self.opt.device.upper()}")
        self.log(f"  输出目录: {self.opt.output_dir}")
        self.log("  提示：点击列表或视频中的车辆框可切换追踪（再次点击取消）")

    # ================================================================
    # UI 构建
    # ================================================================
    def _mk_val(self, txt="--"):
        lb = QLabel(txt)
        lb.setStyleSheet("color:#4af0a0;font-size:15px;font-weight:bold;")
        return lb

    def _build_ui(self):
        cw   = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(5)

        # 标题栏
        title = QLabel("基于深度学习的多目标车辆检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        title.setStyleSheet(
            "color:#4a7cff;padding:8px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a1d2e,stop:.5 #1e2445,stop:1 #1a1d2e);"
            "border-bottom:1px solid #2e3248;"
        )
        root.addWidget(title)

        sp = QSplitter(Qt.Horizontal)
        sp.setHandleWidth(4)

        # ── 左侧：视频区 ────────────────────────────────────────────
        vg = QGroupBox("实时视频观察界面  （点击视频中车辆框可切换追踪该目标）")
        vl = QVBoxLayout(vg)
        vl.setContentsMargins(4, 14, 4, 4)
        self.video_lbl = ClickableVideoLabel()
        self.video_lbl.vehicle_clicked.connect(self._on_veh_click)
        vl.addWidget(self.video_lbl)
        sp.addWidget(vg)

        # ── 右侧面板 ────────────────────────────────────────────────
        ip = QWidget()
        ip.setMaximumWidth(320)
        ip.setMinimumWidth(190)
        il = QVBoxLayout(ip)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(5)

        # 当前帧信息
        fg = QGroupBox("当前帧信息")
        gl = QGridLayout(fg)
        gl.setSpacing(5)
        rows = [
            ("目标数:",   "_lv_count", "0"),
            ("当前帧:",   "_lv_frame", "0"),
            ("处理FPS:",  "_lv_fps",   "--"),
            ("用时:",     "_lv_elap",  "0s"),
            ("追踪ID:",   "_lv_track", "无"),
            ("车辆类型:", "_lv_type",  "--"),
            ("置信度:",   "_lv_conf",  "--"),
            ("位置边框:", "_lv_bbox",  "--"),
        ]
        for r, (lbl, attr, val) in enumerate(rows):
            gl.addWidget(QLabel(lbl), r, 0)
            lb = self._mk_val(val)
            if attr == "_lv_track":
                lb.setStyleSheet("color:#f0a04a;font-size:13px;font-weight:bold;")
            if attr == "_lv_bbox":
                lb.setStyleSheet("color:#8892aa;font-size:11px;")
            setattr(self, attr, lb)
            gl.addWidget(lb, r, 1)
        il.addWidget(fg)

        # 全局统计
        sg = QGroupBox("全局统计")
        sl = QGridLayout(sg)
        sl.setSpacing(5)
        sl.addWidget(QLabel("累计统计车辆:"), 0, 0)
        self._lv_total = self._mk_val("0")
        sl.addWidget(self._lv_total, 0, 1)
        sl.addWidget(QLabel("视频源:"), 1, 0)
        self._lv_src = QLabel("未选择")
        self._lv_src.setStyleSheet("color:#8892aa;font-size:11px;")
        self._lv_src.setWordWrap(True)
        sl.addWidget(self._lv_src, 1, 1)
        sl.addWidget(QLabel("设备:"), 2, 0)
        self._lv_dev = QLabel(self.opt.device.upper())
        self._lv_dev.setStyleSheet("color:#4af0a0;font-size:13px;font-weight:bold;")
        sl.addWidget(self._lv_dev, 2, 1)
        sl.addWidget(QLabel("播放模式:"), 3, 0)
        self._lv_mode = QLabel("--")
        self._lv_mode.setStyleSheet("color:#f0a04a;font-size:11px;")
        sl.addWidget(self._lv_mode, 3, 1)
        il.addWidget(sg)

        # 已检测车辆列表
        tg = QGroupBox("已检测车辆  （点击选中追踪 / 再次点击取消，支持多选≥3）")
        tl = QVBoxLayout(tg)
        tl.setContentsMargins(4, 14, 4, 4)
        tl.setSpacing(4)
        self.veh_list_widget = QListWidget()
        self.veh_list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        self.veh_list_widget.setMinimumHeight(90)
        self.veh_list_widget.setMaximumHeight(210)
        self.veh_list_widget.itemClicked.connect(self._on_vlist_click)
        tl.addWidget(self.veh_list_widget)

        self.btn_clear = QPushButton("✕ 清除所有追踪")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.clicked.connect(self._clear_track)
        self.btn_clear.setEnabled(False)
        self.btn_clear.setFixedHeight(26)
        self.btn_clear.setStyleSheet(
            "QPushButton{padding:2px 8px;font-size:11px;min-width:50px;}"
            "QPushButton#btn_clear{background:#22223a;border-color:#44447a;color:#8888cc;}"
        )
        tl.addWidget(self.btn_clear)
        il.addWidget(tg)

        # 系统日志
        lgg = QGroupBox("系统日志")
        ll  = QVBoxLayout(lgg)
        ll.setContentsMargins(4, 14, 4, 4)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        ll.addWidget(self.log_box)
        il.addWidget(lgg)

        sp.addWidget(ip)
        sp.setStretchFactor(0, 4)
        sp.setStretchFactor(1, 1)
        root.addWidget(sp, 1)

        # ── 控制行1 ─────────────────────────────────────────────────
        c1 = QHBoxLayout()
        c1.setSpacing(8)
        c1.addWidget(QLabel("视频源:"))
        self.combo = QComboBox()
        self.combo.addItems(["摄像头 (0)", "选择视频文件..."])
        self.combo.currentIndexChanged.connect(self._on_src_change)
        c1.addWidget(self.combo)

        self.btn_start = QPushButton("▶ 开始检测/标注")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start)
        c1.addWidget(self.btn_start)

        self.btn_redetect = QPushButton("↺ 重新检测")
        self.btn_redetect.setObjectName("btn_redetect")
        self.btn_redetect.clicked.connect(self._redetect)
        self.btn_redetect.setToolTip("删除已渲染视频，强制重新用AI检测")
        c1.addWidget(self.btn_redetect)

        self.btn_stop = QPushButton("■ 结束运行")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        c1.addWidget(self.btn_stop)

        c1.addStretch()
        self._lv_mstatus = QLabel("模型: 未加载")
        self._lv_mstatus.setStyleSheet("color:#f0a04a;font-size:12px;")
        c1.addWidget(self._lv_mstatus)
        root.addLayout(c1)

        # ── 控制行2 ─────────────────────────────────────────────────
        c2 = QHBoxLayout()
        c2.setSpacing(8)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setObjectName("btn_play")
        self.btn_play.clicked.connect(self._play)
        self.btn_play.setEnabled(False)
        c2.addWidget(self.btn_play)

        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_pause.setEnabled(False)
        c2.addWidget(self.btn_pause)

        self.chk_loop = QCheckBox("循环播放")
        self.chk_loop.setChecked(True)
        self.chk_loop.stateChanged.connect(self._on_loop)
        c2.addWidget(self.chk_loop)
        c2.addStretch()
        root.addLayout(c2)

        # ── 进度条 ──────────────────────────────────────────────────
        pr = QHBoxLayout()
        pr.setSpacing(8)
        pr.addWidget(QLabel("进度:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(lambda: setattr(self, '_drag', True))
        self.slider.sliderReleased.connect(self._on_slide_rel)
        pr.addWidget(self.slider)
        self._lv_prog = QLabel("0 / 0")
        self._lv_prog.setFixedWidth(100)
        self._lv_prog.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pr.addWidget(self._lv_prog)
        root.addLayout(pr)

    # ================================================================
    # 配置加载
    # ================================================================
    def _load_config(self):
        paths = [
            r"D:\python_project\yolov8_car\Vehicle-tracking-main\settings\config.yml",
            r"D:\biyesheji\main_diyibang\main\Vehicle-tracking-main\settings\config.yml",
            r"D:\biyesheji\main\main\Vehicle-tracking-main\settings\config.yml",
            "../../settings/config.yml",
            os.path.join(os.getcwd(), "settings/config.yml"),
        ]
        for p in paths:
            try:
                return OPT(os.path.abspath(p))
            except Exception:
                pass
        # 找不到配置文件则自动生成
        p = os.path.join(os.getcwd(), "settings/config.yml")
        ensure_dir(os.path.dirname(p))
        with open(p, 'w') as f:
            yaml.dump({"device": "cpu", "classes": [0, 1, 2, 3, 5, 7],
                       "conf_thres": 0.25, "iou_thres": 0.5}, f)
        return OPT(p)

    # ================================================================
    # 视频源选择
    # ================================================================
    def _on_src_change(self, idx):
        if idx == 1:
            p, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", get_video_start_dir(),
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.ts);;所有文件 (*.*)"
            )
            if p:
                self.combo.setItemText(1, os.path.basename(p))
                self.combo._vpath = p
                self._current_video_path = p
                self.log(f"已选择视频: {os.path.basename(p)}")
                anno = load_existing_annotations(p, self.opt.output_dir)
                if anno and anno.get('frames'):
                    self.log("  ✓ 找到 JSON 标注，点击开始将 JSON 播放（支持多选追踪）")
                    self._lv_mode.setText("JSON标注播放")
                else:
                    rendered = get_rendered_video_path(p, self.opt.output_dir)
                    if os.path.exists(rendered):
                        self.log("  ✓ 找到已渲染视频，点击开始将直接播放")
                        self._lv_mode.setText("已渲染视频")
                    else:
                        self.log("  ℹ 未找到标注，点击开始将进行 AI 检测")
                        self._lv_mode.setText("全新检测")
            else:
                self.combo.setCurrentIndex(0)

    def _get_src(self):
        if self.combo.currentIndex() == 0:
            return 0
        return getattr(self.combo, '_vpath', None)

    # ================================================================
    # 启动 / 停止
    # ================================================================
    def _start(self, force_redetect=False):
        if self.track_thread and self.track_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先结束当前运行！")
            return

        src = self._get_src()
        if src is None:
            QMessageBox.warning(self, "提示", "请先选择视频文件！")
            return

        if force_redetect and isinstance(src, str):
            rendered = get_rendered_video_path(src, self.opt.output_dir)
            if os.path.exists(rendered):
                try:
                    os.remove(rendered)
                    self.log("✓ 已删除旧渲染视频，将重新 AI 检测")
                except Exception as e:
                    self.log(f"删除旧视频失败: {e}")
            else:
                self.log("未找到旧渲染视频（将直接重新检测）")

        self._stop_thread(keep_video_path=force_redetect)

        if force_redetect and isinstance(src, str):
            self.combo.setItemText(1, os.path.basename(src))
            self.combo._vpath = src
            self.combo.setCurrentIndex(1)
            self._current_video_path = src

        # 重置追踪状态
        self.tracked_ids.clear()
        self.all_seen_vehicles.clear()
        self.veh_list_items.clear()
        self.veh_list_widget.clear()
        self._lv_track.setText("无")
        self.btn_clear.setEnabled(False)

        name = "摄像头(0)" if src == 0 else os.path.basename(str(src))
        self._lv_src.setText(name)
        self.log(f"─── 开始: {name} {'[强制重检]' if force_redetect else ''} ───")

        self.track_thread = TrackingThread(
            src, self.opt,
            load_existing=True,
            force_redetect=force_redetect,
        )
        self.track_thread.frame_ready.connect(self._on_frame)
        self.track_thread.status_msg.connect(self._on_status)
        self.track_thread.truly_finished.connect(self._on_finished)
        self.track_thread.frame_changed.connect(self._on_fc)
        self.track_thread.model_loading.connect(self._on_ml)
        self.track_thread.set_loop(self.chk_loop.isChecked())
        self.track_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_redetect.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.is_playing = True
        self._lv_mstatus.setText("模型: 加载中...")
        self._lv_mstatus.setStyleSheet("color:#f0a04a;font-size:12px;")
        if force_redetect:
            self._lv_mode.setText("全新检测（重检）")

    def _redetect(self):
        if self.track_thread and self.track_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先点击【结束运行】，再使用重新检测！")
            return
        src = self._get_src()
        if src is None or src == 0:
            src = self._current_video_path
        if not src or not isinstance(src, str):
            QMessageBox.warning(self, "提示", "请先选择一个视频文件，再使用重新检测！")
            return
        self.combo.setItemText(1, os.path.basename(src))
        self.combo._vpath = src
        self.combo.setCurrentIndex(1)
        self._start(force_redetect=True)

    def _stop(self):
        self._stop_thread()
        self.video_lbl.setText("已停止")
        self.log("■ 已停止")

    def _stop_thread(self, keep_video_path=False):
        try:
            if self.track_thread and self.track_thread.isRunning():
                self.track_thread.stop()
                self.track_thread.wait(3000)
        except Exception as e:
            self.log(f"停止错误: {e}")
        finally:
            self.track_thread = None

        self._reset_btns()
        self.slider.setEnabled(False)
        self._lv_prog.setText("0 / 0")
        self.is_playing = False

        if not keep_video_path:
            if hasattr(self.combo, '_vpath'):
                del self.combo._vpath
            self.combo.setCurrentIndex(0)
            self.combo.setItemText(1, "选择视频文件...")
            self._lv_src.setText("未选择")

    def _play(self):
        if self.track_thread and self.track_thread.isRunning():
            self.track_thread.resume()
            self.is_playing = True
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.log("▶ 播放")

    def _pause(self):
        if self.track_thread and self.track_thread.isRunning():
            self.track_thread.pause()
            self.is_playing = False
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.log("⏸ 暂停")

    def _reset_btns(self):
        self.btn_start.setEnabled(True)
        self.btn_redetect.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.combo.setEnabled(True)

    # ================================================================
    # 追踪选中逻辑
    # ================================================================
    def _on_veh_click(self, vid):
        if vid in self.tracked_ids:
            self.tracked_ids.discard(vid)
        else:
            self.tracked_ids.add(vid)
        self._push_selection()

    def _on_vlist_click(self, item):
        gid = item.data(Qt.UserRole)
        if gid is None:
            return
        if gid in self.tracked_ids:
            self.tracked_ids.discard(gid)
        else:
            self.tracked_ids.add(gid)
        self._push_selection()

    def _push_selection(self):
        if self.track_thread and self.track_thread.isRunning():
            self.track_thread.set_selected_ids(self.tracked_ids.copy())

        n = len(self.tracked_ids)
        if n == 0:
            self._lv_track.setText("无")
            self.btn_clear.setEnabled(False)
            self.log("取消所有追踪（显示全部车辆）")
        elif n == 1:
            gid = next(iter(self.tracked_ids))
            self._lv_track.setText(f"ID:{gid}")
            self.btn_clear.setEnabled(True)
            self.log(f"→ 追踪车辆 ID:{gid}")
        else:
            ids_str = ",".join(str(i) for i in sorted(self.tracked_ids))
            self._lv_track.setText(f"{n}辆:{ids_str}")
            self.btn_clear.setEnabled(True)
            self.log(f"→ 追踪 {n} 辆 ID:[{ids_str}]")

        self._sync_veh_list_colors()

    def _clear_track(self):
        self.tracked_ids.clear()
        self._lv_track.setText("无")
        self.btn_clear.setEnabled(False)
        self._sync_veh_list_colors()
        self.log("取消所有追踪")
        if self.track_thread and self.track_thread.isRunning():
            self.track_thread.set_selected_ids(set())

    # ================================================================
    # 车辆列表维护
    # ================================================================
    def _update_veh_list(self, all_seen: dict):
        for gid in sorted(all_seen.keys()):
            if gid in self.veh_list_items:
                continue
            item = QListWidgetItem()
            item.setData(Qt.UserRole, gid)
            f = item.font()
            f.setPointSize(11)
            item.setFont(f)
            self.veh_list_widget.addItem(item)
            self.veh_list_items[gid] = item
        self._sync_veh_list_colors()

    def _sync_veh_list_colors(self):
        for gid, item in self.veh_list_items.items():
            vtype = self.all_seen_vehicles.get(gid, 'unknown')
            bgr   = VEHICLE_COLORS.get(vtype, DEFAULT_COLOR)

            if gid in self.tracked_ids:
                r, g, b = bgr[2], bgr[1], bgr[0]
                item.setBackground(QBrush(QColor(r // 2, g // 2, b // 2)))
                item.setForeground(QBrush(QColor(255, 255, 255)))
                item.setText(f"  ★ {vtype}-{gid}  [追踪中]")
            else:
                item.setBackground(QBrush(QColor(20, 22, 38)))
                item.setForeground(QBrush(QColor(180, 185, 200)))
                item.setText(f"  {vtype}-{gid}")

    # ================================================================
    # 信号槽：来自线程
    # ================================================================
    def _on_ml(self, prog, msg):
        self._lv_mstatus.setText(f"模型:{msg[:28]}")
        if prog >= 100:
            self._lv_mstatus.setStyleSheet("color:#4af0a0;font-size:12px;")

    def _on_status(self, msg):
        self.log(msg)
        if "JSON" in msg and "播放" in msg:
            self._lv_mode.setText("JSON标注播放")
            self._lv_mstatus.setText("模式: JSON回放")
            self._lv_mstatus.setStyleSheet("color:#4af0a0;font-size:12px;")
        elif "已渲染视频" in msg or "直接播放" in msg:
            self._lv_mode.setText("已渲染视频")
            self._lv_mstatus.setText("模式: 直接播放")
            self._lv_mstatus.setStyleSheet("color:#4af0a0;font-size:12px;")
        elif "检测" in msg and "完成" in msg:
            self._lv_mode.setText("检测完成→JSON回放")
            self._lv_mstatus.setText("模型: 已完成")
            self._lv_mstatus.setStyleSheet("color:#4af0a0;font-size:12px;")

    def _on_frame(self, qt_img, fidx, fi):
        try:
            vw = fi.get('video_size', (0, 0))[0] or qt_img.width()
            vh = fi.get('video_size', (0, 0))[1] or qt_img.height()

            pix    = QPixmap.fromImage(qt_img)
            scaled = pix.scaled(self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_lbl.setPixmap(scaled)
            self.video_lbl.set_vehicles(fi.get('vehicles', []), (vw, vh))

            self.cur_fi = fi
            self._lv_frame.setText(str(fidx))
            self._lv_count.setText(str(fi.get('num_vehicles', 0)))
            self._lv_total.setText(str(fi.get('total_vehicles', 0)))
            self._lv_fps.setText(f"{fi.get('fps', 0):.1f}")

            # 更新车辆列表（只在有新车辆时触发）
            all_seen = fi.get('all_seen_vehicles', {})
            if all_seen:
                new_gids = [k for k in all_seen if k not in self.all_seen_vehicles]
                if new_gids:
                    self.all_seen_vehicles.update(all_seen)
                    self._update_veh_list(all_seen)

            # 优先展示追踪车辆信息
            vehs = fi.get('vehicles', [])
            tv   = None
            if self.tracked_ids:
                for v in vehs:
                    if v['global_id'] in self.tracked_ids:
                        tv = v
                        break
            if tv is None and vehs:
                tv = vehs[0]

            if tv:
                self._lv_type.setText(tv.get('type', '--'))
                c = tv.get('confidence')
                self._lv_conf.setText(f"{c:.3f}" if c is not None else "--")
                b = tv['bbox']
                self._lv_bbox.setText(f"({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})")
            else:
                self._lv_type.setText("--")
                self._lv_conf.setText("--")
                self._lv_bbox.setText("--")

        except Exception as e:
            self.log(f"帧更新错误: {e}")

    def _on_fc(self, fidx, total):
        if total > 0:
            self.total_frames = total
            self.slider.setEnabled(True)
            if not self._drag:
                self.slider.blockSignals(True)
                self.slider.setValue(int(fidx * 1000 / total))
                self.slider.blockSignals(False)
            self._lv_prog.setText(f"{fidx} / {total}")
        else:
            self._lv_prog.setText(f"{fidx} / ∞")

    def _on_loop(self, s):
        if self.track_thread:
            self.track_thread.set_loop(s == Qt.Checked)

    def _on_slide_rel(self):
        self._drag = False
        if self.track_thread and self.track_thread.isRunning() and self.total_frames > 0:
            target = int(self.slider.value() * self.total_frames / 1000)
            self.track_thread.seek(target)

    def _tick_elapsed(self):
        if 'elapsed' in self.cur_fi:
            self._lv_elap.setText(f"{self.cur_fi['elapsed']:.1f}s")

    def _on_finished(self):
        self._reset_btns()
        self.slider.setEnabled(False)
        self._lv_prog.setText("0 / 0")
        self.is_playing = False
        self._lv_mstatus.setText("模型: 已完成")
        self._lv_mstatus.setStyleSheet("color:#4af0a0;font-size:12px;")
        self.log("─── 处理完成 ───")

    # ================================================================
    # 日志
    # ================================================================
    def log(self, msg):
        try:
            ts = QDateTime.currentDateTime().toString("hh:mm:ss")
            self.log_box.append(f"[{ts}] {msg}")
            doc = self.log_box.document()
            if doc.blockCount() > 600:
                cur = self.log_box.textCursor()
                cur.movePosition(cur.Start)
                for _ in range(doc.blockCount() - 500):
                    cur.select(cur.BlockUnderCursor)
                    cur.removeSelectedText()
                    cur.deleteChar()
        except Exception:
            pass

    def closeEvent(self, ev):
        self._stop_thread()
        self.timer.stop()
        ev.accept()