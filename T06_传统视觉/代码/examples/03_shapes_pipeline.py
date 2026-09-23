# -*- coding: utf-8 -*-
"""
例程 03：传统三板斧 完整管线（Day3 核心例程）
========================================
目标：从一张图里识别出"红色"和"蓝色"图案，并判断每个的形状（圆/方/三角）
完整管线：HSV 阈值 → 形态学去噪 → 找轮廓 → 多边形逼近判形状 → 画质心

怎么运行：
    python examples/03_shapes_pipeline.py        # 终端打印检测结果，弹"识别结果"窗口

看什么：
    终端打印每个图案的形状+质心；窗口里每个图案被标了文字。

学习重点：
    - 把 02 的阈值 + 形态学 + 轮廓 + 形状判断 串成一条完整管线
    - detect_by_color 这个函数 = 三板斧的封装，改 HSV 范围就能识别别的颜色
    - 末尾 TODO：自己加"绿色"识别（练习改两行）
"""
import os
import cv2
import numpy as np
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_shapes.png")

img = imread(IMG)
out = img.copy()  # 画标注的副本
# 把 BGR 彩色图 转成 HSV（H色相/S饱和/V明度），才能按颜色阈值找东西
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def detect_by_color(hsv_img, lower, upper, color_name, color_bgr):
    """输入 HSV 范围，返回该颜色下所有图案的 (形状, 质心) 列表"""
    mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))

    # 形态学去噪：先开后闭
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去小白点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 补小洞

    # 找轮廓（RETR_EXTERNAL 只取最外层）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # 过滤小噪点
            continue

        # 质心
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

        # 形状判断：多边形逼近，顶点数
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
        n = len(approx)
        if n == 3:
            shape = "triangle"
        elif n == 4:
            shape = "rectangle"
        else:
            shape = "circle"

        results.append((shape, (cx, cy), color_name, color_bgr))
    return results, mask


# 红色范围
r1, _ = detect_by_color(hsv, [0, 100, 100], [10, 255, 255], "red", (0, 0, 255))
r2, _ = detect_by_color(hsv, [170, 100, 100], [180, 255, 255], "red", (0, 0, 255))
results = r1 + r2

# 蓝色范围
b, _ = detect_by_color(hsv, [100, 120, 80], [130, 255, 255], "blue", (255, 0, 0))
results += b

# 把结果画到图上
for shape, (cx, cy), color_name, bgr in results:
    cv2.circle(out, (cx, cy), 5, (0, 0, 0), -1)               # 质心黑点
    label = f"{color_name} {shape} ({cx},{cy})"
    # 先画一圈黑色粗字当"描边"，再画彩色细字叠上去，
    # 这样不管背景是什么颜色，文字都清晰（避免和物体颜色重叠看不清）
    pos = (cx + 10, cy - 10)
    cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)   # 黑色描边层
    cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)         # 彩色文字层

print(f"共检测到 {len(results)} 个图案：")
for r in results:
    print(" ", r)

cv2.imshow("Shapes Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO 课后练习（鼓励学生自己动手）：
# 1. 加上"绿色"范围，把绿色三角也识别出来
# 2. 加上"黄色"范围，识别黄线（提示：线条形状判断可能要按面积/长宽比）
# 3. 试试用 quiz_04.png（多形状多颜色+噪点），看你写的能不能扛住

