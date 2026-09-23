# -*- coding: utf-8 -*-
"""
进阶例程 05：边缘检测 Canny
========================================
目标：找到图像中"亮度突变"的边界线（轮廓线）
流程：高斯模糊去噪 → Canny 双阈值 → 输出边缘图

怎么运行：
    python examples/05_canny.py        # 弹 4 个窗口对比

看什么：
    - 高斯模糊前后边缘的干净程度差异
    - Canny(50,150) vs Canny(120,200)：阈值高 → 边缘更少更干净

学习重点：
    - 为什么 Canny 前要高斯模糊（去噪，避免把噪点当边缘）
    - Canny 两个阈值是干嘛的（低阈值=弱边缘，高阈值=强边缘）
    - 应用：黑线巡迹、车道线检测的预备
"""
import os
import cv2
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_gray.png")
img = imread(IMG)                            # 读彩色图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度（Canny 输入要灰度图）

# 1. 高斯模糊：先把图变"糊"一点，去掉细微噪声
#    cv2.GaussianBlur(图, 核大小(宽,高), 标准差0=自动)  核越大越模糊
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Canny 边缘检测
#    cv2.Canny(图, 低阈值50, 高阈值150)
#    低于50=丢掉；50~150之间=弱边缘(和强边缘相连才保留)；高于150=强边缘
edges = cv2.Canny(blur, 50, 150)

# 3. 阈值调高 → 更严格，边缘更少（也更干净，噪声被滤掉）
edges_strict = cv2.Canny(blur, 120, 200)

cv2.imshow("Original Gray", gray)
cv2.imshow("GaussianBlur (5x5)", blur)
cv2.imshow("Canny (50,150)", edges)
cv2.imshow("Canny (120,200)", edges_strict)
finish(["Original Gray", "GaussianBlur (5x5)", "Canny (50,150)", "Canny (120,200)"])

# 提示：
# - Canny 前先高斯模糊，边缘更干净
# - 低阈值太小会出一堆噪声边缘，太大又丢细节，要试
