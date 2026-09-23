# -*- coding: utf-8 -*-
"""
进阶例程 04：二值化（把灰度图变成黑白）
========================================
目标：理解"把灰度图变成纯黑白的三种方式"的区别
  - 固定全局阈值  ：自己定一个数，>它就白，<它就黑（最简单，但怕光照变）
  - OTSU 自适应   ：让 OpenCV 自动找一个最好的阈值（适合明暗分明的图）
  - 自适应阈值    ：每个像素用周围一小块区域单独算阈值（抗光照不均）

怎么运行：
    python examples/04_threshold.py        # 弹 1 个交互窗口对比三种方式

看什么：
    程序弹【1 个交互窗口】，按 2x2 网格显示 4 块：
    ┌──────────┬──────────┐
    │ Original │ Fixed=数值│   ← 滑动条实时调
    ├──────────┼──────────┤
    │OTSU(auto)│ Adaptive │
    └──────────┴──────────┘

    Original      → 原灰度图（处理前参照，均匀浅背景+实心深色物体+细线）
    Fixed=数值    → 固定阈值：亮度 >阈值 白，<阈值 黑（上方滑动条可实时调）
    OTSU(auto)    → OTSU：OpenCV 自动找最佳阈值（固定展示）
    Adaptive      → 自适应：每像素看周围 11x11 区域单独算阈值（固定展示）

    对比深色实心圆（左）和实心方块（右），重点看：

    ① 拖动上方滑动条调【固定阈值】：
       - 调到 30~60 之间（如50）→ Fixed 那块只剩圆（圆黑，方块和背景白）
       - 调高 → 圆和方块都黑（都暗于背景200，分不开）
       → 体会"阈值 = 亮度分界线"（圆30 / 方块60 / 背景200）

    ② 看 OTSU：自动找的阈值，结果接近你调出的合适值

    ③ 看 Adaptive（重点）：
       - 实心圆、实心方块：被"吃空"（内部变白，几乎看不见）
       - 但底部那条【细线】：自适应能完整识别出来！（对比明显）
       → 结论：自适应靠"局部对比"，对【大块实心物体】吃空、
         但适合【细线/边缘】目标（车道线、巡迹线）——正是它的价值所在。

学习重点：
    - threshold 的 4 个参数分别是什么；阈值 = 亮度分界线
    - 固定/OTSU 适合"明暗分明的实心物体"（完整干净）
    - 自适应适合"细线/边缘"目标（车道线、巡迹线）和光照不均场景，
      但对实心物体会把内部吃空、只剩圆环
    - 实战取舍：认实心图案用 OTSU/固定阈值；认线/边缘用自适应
"""
import os
import cv2
import numpy as np
from util import setup, imread

HEADLESS = os.environ.get("HEADLESS") == "1"  # 批量/无头测试时自动退出，不弹窗

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_gray.png")
img = imread(IMG)                            # 读彩色图（BGR）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转成灰度图（二值化前先转灰度）

# 1. 固定全局阈值：见下方交互窗口，用滑动条实时调阈值
#    cv2.threshold(灰度图, 阈值, 最大值255, 方式)：亮度>阈值 白，<阈值 黑

# 2. OTSU：阈值传 0，让 OpenCV 自动找最佳阈值
#    方式里加 cv2.THRESH_OTSU 就开启自动；适合"背景亮、物体暗"分明的图
_, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. 自适应阈值：不用一个全局数，每个像素看它周围 11x11 区域单独定阈值
#    cv2.adaptiveThreshold(灰度图, 最大值, 计算方式, 方式, 邻域大小11, 常数2)
th_adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# =========================================================
# 交互演示：阈值调节器 + 三种方式对比（一个窗口搞定）
# 布局（横向并排）：
#   原图 | 固定阈值(实时可拖) | OTSU | 自适应
# 上方一个滑动条，拖动时【固定阈值】实时变，其他两种固定做对比。
# 目的：亲手体会"阈值 = 亮度分界线"
#   圆亮度30、方块亮度60、背景亮度200
#   - 阈值调到 30~60 之间（如50）→ 固定阈值那块只剩圆（圆黑，方块和背景白）
#   - 阈值调高 → 圆和方块都黑；而 OTSU 一直是自动找的、自适应会吃空实心
# 按 q 退出
# =========================================================
if HEADLESS:
    # 无头测试：不弹窗，只验证三种方式的核心结果是否正确
    _, th_now = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    print("HEADLESS 模式：跳过弹窗。三种二值化已计算完成。")
else:
    WINDOW = "Compare & Tune Threshold (drag bar, q to quit)"
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("Threshold", WINDOW, 150, 255, lambda x: None)
    print("拖动滑动条调【固定阈值】，对比 OTSU/自适应。按 q 退出。")
    print("提示：阈值调到 50 左右，固定阈值那块只剩圆；OTSU 自动找；自适应会吃空实心圆")

# 提前算好三种方式，放进对比（自适应和OTSU不变，固定阈值实时更新）
# OTSU 已在上方算出 th_otsu，自适应已算出 th_adaptive

while True:
    if HEADLESS:
        break  # 无头测试：直接结束，不进入交互窗口
    t = cv2.getTrackbarPos("Threshold", WINDOW)
    # 固定阈值用当前滑动条值 t 实时算
    _, th_now = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    # 每块缩小到一半，避免窗口太宽把右边挤出去
    def shrink(x):
        return cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2))

    # 拼成 2x2 网格（总宽 640，保证 4 块都能看到）
    row1 = np.hstack([shrink(gray), shrink(th_now)])       # 原图 | 固定阈值
    row2 = np.hstack([shrink(th_otsu), shrink(th_adaptive)])  # OTSU | 自适应
    show = np.vstack([row1, row2])

    # 给每块加标题
    labels = [("Original", 0, 0), (f"Fixed={t}", 1, 0), ("OTSU(auto)", 0, 1), ("Adaptive", 1, 1)]
    hh = gray.shape[0] // 2   # 每块缩小后的高
    ww = gray.shape[1] // 2   # 每块缩小后的宽
    for label, rx, cx in labels:
        cv2.putText(show, label, (cx * ww + 6, rx * hh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow(WINDOW, show)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()

# 对比结论：
# - 固定阈值：阈值靠试，调对阈值能"只保留一个物体"（如只显示圆）
# - OTSU：自动算，适合背景和前景对比明显
# - 自适应：局部算阈值，光照不均时最稳（做巡迹线识别常用），但对实心物体会吃空
