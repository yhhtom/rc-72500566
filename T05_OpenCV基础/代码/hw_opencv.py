# -*- coding: utf-8 -*-
"""
T05 OpenCV 基础 —— 作业骨架（学生版）

6 个函数，全部是后面做视觉的"日常操作"。
自检（本教案不含自动校验脚本）：自己写几行 print 调用这六个函数，看输出对不对

注意：
  - 图像是 numpy 数组，shape = (高, 宽, 3)，下标顺序是 [y, x] 不是 [x, y]
  - OpenCV 读进来是 BGR 不是 RGB
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------- 1
def load_image(path: str):
    """读入一张图片（BGR），失败返回 None。直接用 cv2.imread 即可。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    try:
        img = cv2.imread(path)
        return img
    except:
        return None
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 2
def to_gray(img: "np.ndarray") -> "np.ndarray":
    """BGR 转灰度图，返回单通道图。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: cv2.cvtColor + COLOR_BGR2GRAY")
    return cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 3
def crop_roi(img: "np.ndarray", x: int, y: int, w: int, h: int) -> "np.ndarray":
    """裁剪矩形区域。注意 numpy 是 [y:y+h, x:x+w]。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: 切片，先 y 后 x")
    return img[y:y+h, x:x+w]
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 4
def resize_keep(img: "np.ndarray", max_side: int = 640):
    """等比例缩放：让长边等于 max_side，短边按比例。返回缩放后的图。

    640x480 且 max_side=320 -> 320x240
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: 算比例 -> cv2.resize")
    a,b=img.shape[:2]
    if(a>b):
        r = max_side/a
    else:
        r = max_side/b
    a*=r
    b*=r
    return cv2.resize(img,(int(b),int(a)))
    # ↑↑↑ 你的代码写在这里 ↑↑↑
cv2.imwrite("out.png",resize_keep(cv2.imread("test.png"),720))
# ---------------------------------------------------------------- 5
def draw_marker(img: "np.ndarray", cx: float, cy: float,
                text: Optional[str] = None) -> "np.ndarray":
    """在图上画质心：红色圆点(r=5) + 十字，可选在右侧写文字。
    返回新图，不要改动传入的原图（先 copy）。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: copy -> cv2.circle / drawMarker / putText")
    img2 = img.copy()
    cv2.circle(img2,(cx,cy),5,(0,0,255),-1)
    cv2.drawMarker(img2, (cx,cy),(0,0,255),cv2.MARKER_CROSS)
    if text is not None:
        cv2.putText(img2,text,(cx+10,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
    return img2
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 6
def read_frames(source, n: int = 10) -> List["np.ndarray"]:
    """读取帧序列的前 n 帧。

    source 可以是：
      - 摄像头编号（整数，如 0）
      - 视频文件路径（字符串）
    返回帧的 list（读不到就返回已读到的部分）。记得 release。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: VideoCapture + 循环 read + release")
    cap=cv2.VideoCapture(source)
    ret=[]
    for i in range(n):
        ok, frame = cap.read()
        if(not ok):
            break
        ret.append(frame)
    cap.release()
    return ret
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 7（选做）
def mask_centroid(mask: "np.ndarray") -> Optional[Tuple[float, float]]:
    """求二值掩膜中白色区域的质心 (cx, cy)；全黑（没有白色）返回 None。

    提示：cv2.moments(mask)，m00 为 0 表示没有白色像素。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: cv2.moments")
    M = cv2.moments(mask)
    if M["m00"]==0:
        return None
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    return cx, cy
    # ↑↑↑ 你的代码写在这里 ↑↑↑

cv2.imwrite("test.png",np.zeros((360,360)))
cv2.imwrite("out.png",draw_marker(cv2.imread("test.png"),180,180))
cv2.imwrite("mask.png",cv2.inRange(cv2.imread("out.png"),(0,0,1),(0,0,255)))