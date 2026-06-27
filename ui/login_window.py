# -*- coding: utf-8 -*-
"""
login_window.py
LoginWindow：系统登录界面
验证通过后打开 MainWindow
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from ui.styles import LOGIN_STYLE


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的多目标车辆检测系统")
        self.resize(520, 400)
        self.setMinimumSize(460, 340)
        self.setStyleSheet(LOGIN_STYLE)

        w    = QWidget()
        self.setCentralWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # 标题
        t = QLabel("基于深度学习的多目标车辆检测系统")
        t.setAlignment(Qt.AlignCenter)
        t.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        t.setStyleSheet("color:#4a7cff;margin-bottom:10px;")
        root.addWidget(t)

        # 表单
        form = QGridLayout()
        form.setSpacing(10)
        form.setColumnStretch(1, 1)

        lu = QLabel("用户名:"); lu.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lp = QLabel("密　码:"); lp.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.u_edit = QLineEdit(placeholderText="请输入用户名")
        self.p_edit = QLineEdit(placeholderText="请输入密码", echoMode=QLineEdit.Normal)
        self.p_edit.returnPressed.connect(self._login)

        form.addWidget(lu, 0, 0); form.addWidget(self.u_edit, 0, 1)
        form.addWidget(lp, 1, 0); form.addWidget(self.p_edit, 1, 1)
        root.addLayout(form)

        # 登录按钮
        btn = QPushButton("登 录")
        btn.clicked.connect(self._login)
        root.addWidget(btn)

        tip = QLabel("")
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet("color:#5a607a;font-size:11px;")
        root.addWidget(tip)

    def _login(self):
        u = self.u_edit.text().strip()
        p = self.p_edit.text().strip()
        if not u or not p:
            QMessageBox.warning(self, "提示", "用户名和密码不能为空！")
            return
        if u == "abcde" and p == "123456":
            QMessageBox.information(self, "提示", "登录成功！")
            QTimer.singleShot(80, self._open_main)
        elif p == "123456":
            QMessageBox.warning(self, "提示", "陌生用户无法登录！")
        else:
            QMessageBox.warning(self, "提示", "用户名或密码错误！")

    def _open_main(self):
        try:
            # 延迟导入，避免循环依赖
            from ui.main_window import MainWindow
            self.mw = MainWindow()
            self.mw.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"主窗口打开失败:\n{e}")