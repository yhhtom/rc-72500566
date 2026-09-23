# -*- coding: utf-8 -*-
"""
例程 02b：HSV 颜色调节器（滑动条实时调阈值）★新手必用
========================================
目标：拖动滑动条，实时看到底哪些像素会被选中，快速锁定颜色的 HSV 范围
（比手抄固定数值更快更直观）

怎么运行：
    python examples/02b_hsv_trackbar.py        # 弹"HSV Tuner"窗口，拖动 6 个滑动条

看什么：
    窗口右边白色区域 = 会被选中的像素。调到"目标全白、背景全黑"就对了。

学习重点：
    - 六个滑动条 = H/S/V 各自的 Min/Max
    - 调好后把 6 个数字抄下来，填进 cv2.inRange 就完成了颜色锁定
    - 口诀：先调 H 锁颜色 → 再调 S 滤灰 → 最后调 V 滤暗
"""
import os
import cv2
import numpy as np
from util import setup, imread

HEADLESS = os.environ.get("HEADLESS") == "1"  # 批量/无头测试时自动退出，不弹窗

setup()

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_shapes.png")
img = imread(IMG)
# 把 BGR 彩色图 转成 HSV（H色相/S饱和/V明度），才能按颜色阈值找东西
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

WINDOW = "HSV Tuner (drag bars, white=selected)"


def nothing(x):
    pass


# 六个滑动条：H/S/V 各自的最小值和最大值
# createTrackbar(名字, 窗口, 初始值, 最大值, 回调)
# 初始值给的是"红色"的推荐起点（demo_shapes 里红圆最容易调），
# 程序一打开就已经选到红色，再微调就能锁住，不用从全范围盲调。
cv2.namedWindow(WINDOW)
cv2.createTrackbar("H Min", WINDOW, 0, 179, nothing)    # H最小值：红色从 0 开始
cv2.createTrackbar("H Max", WINDOW, 10, 179, nothing)   # H最大值：段1到 10（红色两段之一）
cv2.createTrackbar("S Min", WINDOW, 100, 255, nothing)  # S最小值：饱和度≥100，滤掉浅色/灰
cv2.createTrackbar("S Max", WINDOW, 255, 255, nothing)  # S最大值：到255
cv2.createTrackbar("V Min", WINDOW, 100, 255, nothing)  # V最小值：明度≥100，滤掉太暗的
cv2.createTrackbar("V Max", WINDOW, 255, 255, nothing)  # V最大值：到255

# 推荐初始值参考（自己调其他颜色可从这里起步）：
#   红色 : H[0,10]+[170,179]  S[100,255]  V[100,255]  （注意红色要两段，这里先演示段1）
#   绿色 : H[35,85]  S[100,255]  V[100,255]
#   蓝色 : H[100,130] S[100,255]  V[100,255]
#   黄色 : H[20,35]  S[100,255]  V[100,255]
# 记不住就拖动滑动条，看白色区域实时变化来试。

print("拖动滑动条，白色区域就是被选中的像素。按 q 退出。")
print("调好后把 6 个数字抄下来，填进 cv2.inRange 即可。")

while True:
    if HEADLESS:
        break  # 无头测试：只验证逻辑，不进入交互
    h_min = cv2.getTrackbarPos("H Min", WINDOW)
    h_max = cv2.getTrackbarPos("H Max", WINDOW)
    s_min = cv2.getTrackbarPos("S Min", WINDOW)
    s_max = cv2.getTrackbarPos("S Max", WINDOW)
    v_min = cv2.getTrackbarPos("V Min", WINDOW)
    v_max = cv2.getTrackbarPos("V Max", WINDOW)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # 拼接显示：左边原图，右边掩膜
    show = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    # 显示当前范围文字
    cv2.putText(show, f"H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow(WINDOW, show)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
