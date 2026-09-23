# -*- coding: utf-8 -*-
"""
进阶例程 06：形态学全家桶（腐蚀/膨胀/开/闭/顶帽/黑帽）
========================================
目标：掌握图像形态学处理的各种操作和适用场景
  - 腐蚀/膨胀  ：最基础，白色区域"瘦一圈/胖一圈"
  - 开运算/闭运算：先腐蚀后膨胀 / 先膨胀后腐蚀，去噪/补洞
  - 顶帽/黑帽  ：提取比背景亮的部分 / 比背景暗的部分

怎么运行：
    python examples/06_morphology.py        # 弹 7 个窗口对比所有操作

看什么：
    demo_noise_text.png 里撒了很多黑白噪点，看哪种操作能清掉它们。

学习重点：
    - 形态学是对"二值图"(黑白)做的，先要二值化
    - "开运算去噪点，闭运算补空洞" 这句口诀
    - kernel(核) 大小决定影响范围
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_noise_text.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化：把"黑色物体"变成白色(255)、背景变黑(0)
# cv2.threshold(灰度图, 阈值127, 最大值, 方式INV=黑白反过来)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 结构元素(核)：3x3 全1的方块，决定每次操作"看多大范围"
kernel = np.ones((3, 3), np.uint8)

# 腐蚀/膨胀：cv2.erode(图, 核, iterations次数=1) 核越大、次数越多，效果越强
erode = cv2.erode(binary, kernel, iterations=1)          # 腐蚀：白色区域缩小，去小白点
dilate = cv2.dilate(binary, kernel, iterations=1)        # 膨胀：白色区域扩大，补小洞

# 开/闭/顶帽/黑帽：cv2.morphologyEx(图, 操作类型, 核)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 开运算 = 先腐蚀后膨胀 → 去噪点
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算 = 先膨胀后腐蚀 → 补空洞
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)  # 顶帽 = 原图 - 开运算 → 亮区域
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)  # 黑帽 = 闭运算 - 原图 → 暗区域

cv2.imshow("Binary (INV)", binary)
cv2.imshow("Erode (remove noise)", erode)
cv2.imshow("Dilate (fill hole)", dilate)
cv2.imshow("Opening (erode+dilate)", opening)
cv2.imshow("Closing (dilate+erode)", closing)
cv2.imshow("TopHat (bright)", tophat)
cv2.imshow("BlackHat (dark)", blackhat)
finish(["Binary (INV)", "Erode (remove noise)", "Dilate (fill hole)",
        "Opening (erode+dilate)", "Closing (dilate+erode)", "TopHat (bright)", "BlackHat (dark)"])

# 记忆：
# 开运算 → 去白色噪点；闭运算 → 补黑色空洞
# 顶帽 → 提取比背景亮的部分；黑帽 → 提取比背景暗的部分
