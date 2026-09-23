# -*- coding: utf-8 -*-
"""
进阶例程 08：霍夫圆检测 HoughCircles
========================================
目标：直接找出图像里的"圆形"，返回圆心坐标和半径
  - 比"数顶点判形状"更直接，专门用来找圆
  - 应用：靶心识别、瓶盖检测、硬币识别

怎么运行：
    python examples/08_hough_circles.py        # 弹窗口，把检测到的圆圈出来

看什么：
    demo_lines_circles.png 里有 3 个圆，看能不能都找到并标出圆心半径。

学习重点：
    - 霍夫圆对噪声敏感，先高斯模糊
    - 6 个参数的调参逻辑（下面代码有注释），改参数看检出数量变化
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_lines_circles.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 霍夫圆对噪声敏感，先高斯模糊
blur = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,          # 累加器分辨率（1.2 常用）
    minDist=50,      # 两圆最小圆心距（小于它会被合并）
    param1=100,      # Canny 高阈值
    param2=30,       # 圆心累加器阈值（越小越容易误检）
    minRadius=10,    # 最小半径
    maxRadius=150,   # 最大半径
)

out = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(out, (x, y), r, (0, 0, 255), 3)   # 外圈
        cv2.circle(out, (x, y), 2, (0, 255, 0), -1)  # 圆心
        cv2.putText(out, f"r={r}", (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)  # 黑色描边，避免和背景重叠
        cv2.putText(out, f"r={r}", (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
print(f"检测到 {0 if circles is None else len(circles[0])} 个圆")

cv2.imshow("HoughCircles", out)
finish()

# 参数调优口诀：
# param2 调小 → 检出更多圆（也更多假圆）；调大 → 更严
# minDist 调大 → 避免小圆互相干扰
# minRadius/maxRadius → 限定圆的大小范围，能滤掉干扰
