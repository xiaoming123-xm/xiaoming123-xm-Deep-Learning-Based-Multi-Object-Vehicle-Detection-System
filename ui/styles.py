# -*- coding: utf-8 -*-
"""
styles.py
主窗口 QSS 样式字符串，统一管理所有控件的外观
"""

MAIN_STYLE = """
QMainWindow,QWidget{
    background:#1a1d2e; color:#c8ccd8;
    font-family:"Microsoft YaHei","Segoe UI",sans-serif;
}
QGroupBox{
    color:#6878aa; border:1px solid #2e3248; border-radius:6px;
    margin-top:12px; font-size:12px; padding-top:4px;
}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;}
QPushButton{
    background:#252a3c; color:#c8ccd8;
    border:1px solid #3a3f55; padding:7px 14px;
    font-size:13px; border-radius:5px; min-width:76px;
}
QPushButton:hover  {background:#2e3450;border-color:#4a5070;}
QPushButton:pressed{background:#1e2235;}
QPushButton:disabled{color:#454a5a;border-color:#252a3c;}
#btn_start{background:#1a3d26;border-color:#267a3e;color:#55cc70;}
#btn_start:hover{background:#1e4a2c;}
#btn_redetect{background:#1a2d3d;border-color:#1f5a7a;color:#3aadcc;}
#btn_redetect:hover{background:#1e3550;}
#btn_stop {background:#3d1a1a;border-color:#7a2626;color:#cc5555;}
#btn_stop:hover{background:#4a2020;}
#btn_play {background:#1a2d3d;border-color:#266a7a;color:#55aacc;}
#btn_pause{background:#2d2d1a;border-color:#6a6a26;color:#cccc55;}
#btn_clear{background:#22223a;border-color:#44447a;color:#8888cc;}
QComboBox{
    background:#252a3c;color:#c8ccd8;
    border:1px solid #3a3f55;padding:5px 8px;
    border-radius:4px;font-size:13px;
}
QComboBox::drop-down{border:none;}
QComboBox QAbstractItemView{
    background:#252a3c;color:#c8ccd8;
    selection-background-color:#3a4060;
}
QTextEdit{
    background:#0d0f18;color:#66ee66;
    border:1px solid #2e3248;
    font-family:Consolas,monospace;font-size:11px;
}
QListWidget{
    background:#0d0f18;color:#c8ccd8;
    border:1px solid #2e3248;
    font-size:12px;
}
QListWidget::item{
    padding:5px 8px;
    border-bottom:1px solid #1a1d2e;
}
QListWidget::item:hover{background:#1e2440;}
QSlider::groove:horizontal{
    border:none;height:6px;
    background:#2e3248;border-radius:3px;
}
QSlider::sub-page:horizontal{background:#4a7cff;border-radius:3px;}
QSlider::handle:horizontal{
    background:#6a9cff;border:none;
    width:14px;height:14px;margin:-4px 0;border-radius:7px;
}
QSlider::handle:horizontal:hover{background:#8abcff;}
QCheckBox{color:#8892aa;spacing:6px;}
QCheckBox::indicator{
    width:14px;height:14px;border:1px solid #3a3f55;
    border-radius:3px;background:#252a3c;
}
QCheckBox::indicator:checked{background:#4a7cff;border-color:#4a7cff;}
QSplitter::handle{background:#2e3248;}
"""

LOGIN_STYLE = """
QMainWindow,QWidget{background:#1a1d2e;}
QLabel{color:#c8ccd8;font-size:13px;}
QLineEdit{
    padding:9px 12px;font-size:13px;
    border:1px solid #3a3f55;border-radius:6px;
    background:#252a3c;color:#e0e4f0;
}
QLineEdit:focus{border:1px solid #4a7cff;}
QPushButton{
    padding:10px;font-size:14px;font-weight:bold;
    background:#4a7cff;color:white;
    border:none;border-radius:6px;
}
QPushButton:hover{background:#5a8cff;}
QPushButton:pressed{background:#3a6cee;}
"""