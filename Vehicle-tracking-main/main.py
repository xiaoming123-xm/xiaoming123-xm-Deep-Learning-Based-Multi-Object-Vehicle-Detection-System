# -*- coding: utf-8 -*-
"""
main.py
程序入口：初始化 Qt 应用，显示登录窗口
支持 CLI 模式（传入 cli 参数时走命令行检测）
"""

import sys
import os

# ── 环境变量（必须在 import torch 之前设置）──────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"]   = "TRUE"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]    = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]     = "1"

# ── 加入 yolov5 库路径（如果存在）───────────────────────────────────
lib_path = os.path.abspath(os.path.join('infrastructure', 'yolov5'))
if lib_path not in sys.path:
    sys.path.append(lib_path)

import warnings
warnings.filterwarnings("ignore")

import torch

try:
    import torch.backends.cudnn as cudnn
    if torch.cuda.is_available():
        cudnn.benchmark = True
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# ── 尝试导入原始 CLI 追踪器（可选，无则跳过）───────────────────────
try:
    from infrastructure.handlers.track import Tracker as OriginalTracker
except ImportError:
    class OriginalTracker:
        def __init__(self, config_path): pass
        def detect(self): pass


def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

        if len(sys.argv) > 1 and sys.argv[1] == "cli":
            # ── CLI 模式：命令行直接检测 ─────────────────────────────
            tracker = OriginalTracker("../../settings/config.yml")
            with torch.no_grad():
                tracker.detect()
        else:
            # ── GUI 模式：显示登录界面 ───────────────────────────────
            from ui.login_window import LoginWindow
            login = LoginWindow()
            login.show()
            sys.exit(app.exec_())

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()