# -*- coding: utf-8 -*-
"""
T06 传统视觉三板斧 —— 作业骨架（学生版）

三板斧：HSV 颜色阈值 → 形态学去噪 → 轮廓筛选 + 质心
四个函数对应四张图（images/ 或 assets/ 下的 quiz_0X.png）。

自检（本教案不含自动校验脚本）：自己写几行代码跑一遍 images/quiz_0X.png，
把算出来的坐标打印出来，和讲义里的真值对一下（容差 8 px）

调试利器（强烈推荐先玩这个）：
    python examples/02b_hsv_trackbar.py
    拖动滑动条找到合适的 HSV 范围，再把数值抄进这里。
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


RED_HI1 = (180,255,255)
RED_LO1 = (170,100,50)
RED_HI2 = (10,255,255)
RED_LO2 = (0,100,50)
BLUE_HI = (130,255,255)
BLUE_LO = (100,100,50)
GREEN_HI = (85,255,255)
GREEN_LO = (35,100,50)
OUT = "/home/pd20w/Desktop/robocon_hw/T06_传统视觉/代码/out/"
IN = "/home/pd20w/Desktop/robocon_hw/T06_传统视觉/代码/images/"
# ---------------------------------------------------------------- 工具（已给）
def _morph(mask, k=5):
    """开运算去噪 + 闭运算补洞"""
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _centroid(cnt) -> Optional[Tuple[float, float]]:
    """算轮廓质心，退化轮廓返回 None"""
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _vertices(cnt) -> int:
    """多边形逼近后的顶点数（3=三角 4=四边 >=8≈圆）"""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    return len(approx)


def _in_range(img, lo, hi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

# ---------------------------------------------------------------- 通用小工具
def _pts(cnt):
    """轮廓 -> (N, 2) float 点集"""
    return cnt.reshape(-1, 2).astype(np.float32)


def _angle(p0, p1, p2):
    """返回 p1 处的夹角（度），p0-p1-p2"""
    v1, v2 = p0 - p1, p2 - p1
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    c = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _approx(cnt, eps_ratio=0.02):
    """多边形逼近，返回 (K, 2) 点集"""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
    return _pts(approx)


def _is_circle(cnt, min_area=100, circularity_th=0.7) -> bool:
    """判断轮廓是否接近圆形。

    判据：圆形度 = 4πA / P²，完美圆 = 1。
    """
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False

    circularity = 4 * np.pi * area / (peri * peri)
    return circularity > circularity_th


def _is_rectangle(cnt, min_area=100, angle_tol=15, eps_ratio=0.02) -> bool:
    """判断轮廓是否接近矩形（含正方形）。

    判据：逼近后 4 个顶点、凸、四个角都接近 90°。
    """
    if cv2.contourArea(cnt) < min_area:
        return False

    pts = _approx(cnt, eps_ratio)
    if len(pts) != 4:
        return False
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return False

    # 四个内角都要接近 90°
    for i in range(4):
        a = _angle(pts[i - 1], pts[i], pts[(i + 1) % 4])
        if abs(a - 90) > angle_tol:
            return False
    return True


def _is_square(cnt, min_area=100, side_tol=0.1,
               angle_tol=15, eps_ratio=0.02) -> bool:
    """判断轮廓是否接近正方形。

    在矩形基础上，额外要求四边等长（最长/最短 < 1+side_tol）。
    """
    if not _is_rectangle(cnt, min_area, angle_tol, eps_ratio):
        return False

    pts = _approx(cnt, eps_ratio)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    if min(sides) == 0:
        return False

    return (max(sides) / min(sides)) < (1 + side_tol)

# ---------------------------------------------------------------- 1
def detect_red_circle(img) -> Optional[Tuple[float, float]]:
    """quiz_01：找出图中唯一的红色圆，返回质心 (cx, cy)；找不到返回 None。

    提示：红色的 H 在 0 附近，需要两段阈值 [0,10] 与 [170,180] 合并。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: HSV 双阈值 -> 形态学 -> 最大轮廓 -> 质心")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img,RED_LO1,RED_HI1)
    mask2 = cv2.inRange(img,RED_LO2,RED_HI2)
    mask = cv2.bitwise_or(mask1,mask2)
    mask = _morph(mask)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxs = 0
    maxcnt = None
    for cnt in cnts:
        if not _is_circle(cnt):
            continue
        if len(cnt[0])>maxs:
            maxs = len(cnt[0])
            maxcnt = cnt
    return _centroid(maxcnt)
    # ↑↑↑ 你的代码写在这里 ↑↑↑
img = cv2.imread(IN+"quiz_01.png")
x,y = detect_red_circle(img)
print(x,y)
out = cv2.drawMarker(img,(int(x),int(y)),(0,255,0),cv2.MARKER_CROSS)
cv2.imshow("Quiz01", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------- 2
def detect_blue_rect(img) -> Optional[Tuple[float, float]]:
    """quiz_02：找出蓝色正方形，返回质心。蓝色 H 约 100~130。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img,BLUE_LO,BLUE_HI)
    mask = _morph(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxs = 0
    maxcnt = None
    print(len(cnts))
    for cnt in cnts:
        if not _is_square(cnt):
            continue
        if len(cnt[0])>maxs:
            maxs=len(cnt[0])
            maxcnt = cnt
    return _centroid(maxcnt)
    # ↑↑↑ 你的代码写在这里 ↑↑↑
img = cv2.imread(IN+"quiz_02.png")
x,y = detect_blue_rect(img)
out = cv2.drawMarker(img,(int(x),int(y)),(0,255,0),cv2.MARKER_CROSS)
cv2.imshow("Quiz02", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------- 3
def detect_green_triangle(img) -> Optional[Tuple[float, float]]:
    """quiz_03：图中有绿三角 + 红圆干扰 + 蓝块干扰。
    只返回绿色三角形的质心（用顶点数 == 3 筛选）。绿色 H 约 35~85。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: 找出所有轮廓后按顶点数过滤")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img,GREEN_LO,GREEN_HI)
    mask = _morph(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxs = 0
    maxcnt = None
    for cnt in cnts:
        if len(_approx(cnt)) != 3:
            continue
        if len(cnt[0])>maxs:
            maxs=len(cnt[0])
            maxcnt = cnt
    return _centroid(maxcnt)
    # ↑↑↑ 你的代码写在这里 ↑↑↑
img = cv2.imread(IN+"quiz_03.png")
x,y = detect_green_triangle(img)
out = cv2.drawMarker(img,(int(x),int(y)),(0,0,255),cv2.MARKER_CROSS)
cv2.imshow("Quiz03", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---------------------------------------------------------------- 4
def detect_red_targets(img) -> List[Tuple[float, float]]:
    """quiz_04：找出图中所有红色目标的质心，按 x 升序返回。

    注意：橙色圆 (H≈19) 不是红色，别把它算进来 —— 把 H 上限开太大就会误检。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: 找所有红色轮廓 -> 质心 -> sorted(key=lambda p: p[0])")
    p = []
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img,RED_LO1,RED_HI1)
    mask2 = cv2.inRange(img,RED_LO2,RED_HI2)
    mask = cv2.bitwise_or(mask1,mask2)
    mask = _morph(mask)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(cnts))
    for cnt in cnts:
        p.append(_centroid(cnt))
    sorted(p,key=lambda p: p[0])
    return p
    # ↑↑↑ 你的代码写在这里 ↑↑↑
img = cv2.imread(IN+"quiz_04.png")
for x,y in detect_red_targets(img):
    cv2.drawMarker(img,(int(x),int(y)),(0,255,0),cv2.MARKER_CROSS)
cv2.imshow("Quiz04", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
