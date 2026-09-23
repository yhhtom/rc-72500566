# -*- coding: utf-8 -*-
"""
进阶例程 07：霍夫直线检测 HoughLinesP
========================================
目标：从边缘图里找出"直线"，返回每段的端点和角度
  - 比轮廓更"聪明"：专门找笔直的线（比如跑道线、黑板框）
  - 还能算出角度，可用来判断"线是横是竖是斜"

怎么运行：
    python examples/07_hough_lines.py        # 弹窗口，图上标出检测到的线和角度

看什么：
    demo_lines_circles.png 里有 3 条直线，看它能不能都找出来。
    每条直线会被画成红色，起点上方标注一个角度值，如：
      "45deg" = 这条线相对水平方向倾斜 45°
      "0deg"  = 横线   "90deg" = 竖线   "负值" = 从左上往右下斜
    通过角度能一眼判断线的方向（横/竖/斜），做车道线检测很有用。

学习重点：
    - 霍夫直线前要先做 Canny 边缘检测（Hough 吃的是边缘图）
    - HoughLinesP 的几个参数分别调什么
    - 角度 = np.degrees(np.arctan2(dy, dx))：判断线的倾斜方向
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_lines_circles.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 先做 Canny 边缘检测，霍夫直线在"边缘图"上找线
edges = cv2.Canny(gray, 50, 150)

# HoughLinesP 概率霍夫直线检测，参数：
#   rho=1           累加器距离精度(像素)，1 即可
#   theta=π/180     累加器角度精度(弧度)，1 度
#   threshold=50    多少个"点"算一条线（太小→噪声多，太大→漏线）
#   minLineLength=30 最短线长(像素)，太短的线不要
#   maxLineGap=10   断线最大间隔(像素)，虚线也能连起来
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                        threshold=50, minLineLength=30, maxLineGap=10)

out = img.copy()
if lines is not None:
    # 兼容 OpenCV 4.x 与 5.x：4.x 返回 (N,1,4)，5.x 返回 (N,4)，统一拍成 (N,4)
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # 计算角度（度）：arctan2 算这条线相对水平方向的斜率角，再转成角度
        #   0°=横线  90°=竖线  正=右下斜  负=左下斜
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 把角度标注在线段起点上方，如 "45deg"
        cv2.putText(out, f"{ang:.0f}deg", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)  # 黑色描边
        cv2.putText(out, f"{ang:.0f}deg", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
print(f"检测到 {0 if lines is None else len(lines)} 条直线")

cv2.imshow("Edges (Canny)", edges)
cv2.imshow("Detected Lines", out)
finish(["Edges (Canny)", "Detected Lines"])

# 参数说明：
# rho/theta：累加器的距离和角度精度（1 像素 / 1 度即可）
# threshold：多少个点才算一条线（越小越容易出线，但噪声多）
# minLineLength：最短线长；maxLineGap：断线最大间隔
