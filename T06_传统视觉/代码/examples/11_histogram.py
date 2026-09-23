# -*- coding: utf-8 -*-
"""
进阶例程 11：直方图 + 直方图均衡化
========================================
目标：
  1. 画灰度直方图，看懂"图像亮度怎么分布"
  2. 直方图均衡化，提升低对比度图像的清晰度

怎么运行：
    python examples/11_histogram.py        # 弹 4 个窗口：原图/直方图/均衡化图/均衡化直方图

看什么：
    弹 4 个窗口，重点是两个直方图（带坐标轴）：
      - x 轴 = 亮度 0~255；y 轴 = 像素数量（柱子越高 = 该亮度像素越多）
      - Histogram（均衡化前）：亮度挤在中间一段（80~160），灰蒙蒙
      - Equalized Histogram（均衡化后）：亮度铺开到 0~255，对比度提升

    怎么读直方图（下面"末尾注释"有更详细的表）：
      1. 看峰的位置：峰在左=图偏暗，在右=图偏亮，在中间=灰蒙蒙
      2. 看峰的高度：越高=那个亮度的像素越多
      3. 对比两幅直方图：均衡化让分布更均匀 = 对比度提升

学习重点：
    - 直方图 = 统计"每个亮度有多少像素"，峰在左=图暗，在右=图亮
    - 均衡化适合低对比度图：把挤在一起的亮度拉开 → 对比度提升
    - 注意：对比已经分明的图（如 demo_gray）均衡化效果差，别用它演示
    - 应用：低光照/雾图增强
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码


def draw_hist(hist, title):
    """把直方图画成带坐标轴的图（更好看、比例更舒服）
    画布 300 高 x 280 宽：左侧留 y 轴、底部留 x 轴标签。
    x 轴 = 亮度 0~255；y 轴 = 像素数量（峰值顶到顶、留一点边）。
    """
    W, H, MARGIN_L, MARGIN_B = 280, 300, 46, 30   # 画布尺寸和边距
    img = np.full((H, W), 255, np.uint8)           # 白画布

    # 直方图区域（去掉边距后剩下的部分）
    plot_w, plot_h = W - MARGIN_L, H - MARGIN_B
    # 峰值归一化到 plot_h 的 95%，留点顶边
    peak = hist.max() if hist.max() > 0 else 1
    scale = (plot_h * 0.95) / peak

    # 画每个亮度等级的柱子（每根柱子约 0.8 像素宽，带一点间距更清爽）
    # 兼容 OpenCV 4.x 与 5.x：4.x 的 calcHist 返回 (256,1)，5.x 返回 (256,)，统一拉平成一维
    hist = np.asarray(hist).reshape(-1)
    for x in range(256):
        h = int(hist[x] * scale)
        x0 = MARGIN_L + int(x * plot_w / 256)
        cv2.line(img, (x0, H - MARGIN_B - h), (x0, H - MARGIN_B), (0, 0, 0), 1)

    # 画坐标轴（两条粗黑线）
    cv2.line(img, (MARGIN_L, MARGIN_B // 2), (MARGIN_L, H - MARGIN_B), (0, 0, 0), 2)  # y轴
    cv2.line(img, (MARGIN_L, H - MARGIN_B), (W - 2, H - MARGIN_B), (0, 0, 0), 2)      # x轴

    # x 轴刻度：0 / 64 / 128 / 192 / 255
    for val in [0, 64, 128, 192, 255]:
        x0 = MARGIN_L + int(val * plot_w / 256)
        cv2.line(img, (x0, H - MARGIN_B), (x0, H - MARGIN_B + 4), (0, 0, 0), 2)
        cv2.putText(img, str(val), (x0 - 12, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "brightness(0-255)", (MARGIN_L + 60, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # 标题
    cv2.putText(img, title, (MARGIN_L, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img


IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_lowcontrast.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. 计算直方图：统计 0~255 每个亮度等级各有多少像素
#    cv2.calcHist([图], [通道0], None, [256个桶], [范围0~256])
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_img = draw_hist(hist, "Original Histogram")

# 2. 直方图均衡化：重新分配亮度，让亮暗拉开、对比度提升
equ = cv2.equalizeHist(gray)
hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])
hist_equ_img = draw_hist(hist_equ, "Equalized Histogram")

cv2.imshow("Original Gray", gray)
cv2.imshow("Histogram", hist_img)
cv2.imshow("Equalized", equ)
cv2.imshow("Histogram Equalized", hist_equ_img)
finish(["Original Gray", "Histogram", "Equalized", "Histogram Equalized"])

# =====================================================
# 直方图怎么看？（对照弹出来的两个直方图窗口）
# =====================================================
# 1. 看"横轴"（x轴=亮度0~255）：
#     柱子集中在左边(0~80)  → 图偏暗
#     柱子集中在右边(176~255) → 图偏亮
#     柱子集中在中间(80~176) → 图灰蒙蒙（低对比度）
#
# 2. 看"柱子的高度"（y轴=像素数量）：
#     某个亮度柱子越高 → 该亮度的像素越多
#     （比如背景占比大，背景亮度的柱子就最高）
#
# 3. 看"分布范围"：
#     分布窄（只占一小段） → 对比度低，细节看不清
#     分布宽（铺满 0~255） → 对比度高，黑白分明
#
# 4. 对比"均衡化前后"两幅直方图：
#     均衡化前：亮度挤在中间一段（demo_lowcontrast 灰蒙蒙图）
#     均衡化后：亮度铺开到整个 0~255 范围
#     → 这就是"均衡化拉开对比度"：把挤在一起的亮度摊开，
#       原本看不清的明暗细节变得分明。
#
# 一句话总结：
#   直方图 = 图里"每种亮度有多少像素"的统计表。
#   峰在左=暗，在右=亮，在中间=灰蒙蒙；铺得越开=对比度越高。
# =====================================================
