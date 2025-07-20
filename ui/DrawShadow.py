#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：PyStdPlane
@Author  ：htk
@Date    ：2023/9/26 19:24
'''

from PySide6.QtGui import QPen, QFont, QPainter, QColor


class DrawShadow:

    @classmethod
    def draw_shadow(cls, painter: QPainter, text, x, y, font_size):
        # 设置Qpainter
        pen = QPen(QColor(252, 243, 81))
        painter.setPen(pen)
        font = QFont('Arial', font_size)
        painter.setFont(font)

        painter.drawText(x, y, text)
