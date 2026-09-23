# -*- coding: utf-8 -*-
"""
T07 进阶 · 完整实时追踪程序（全部自己写）
========================================
这一题只给需求与思路，**不给代码框架**——它是把 T06/T07 学过的东西
串成"一个真正能用的程序"，也是电赛现场最需要的那种程序。

需求
----
  1. 打开视频源，实时识别**指定颜色**图案的质心，并在画面上实时显示：
       · 质心画十字 + 圆圈，旁边写实时坐标 (x, y)
       · 左上角显示"当前在追什么颜色"与 FPS
  2. 按 1 / 2 / 3 / 4 切换追踪颜色：1=红 2=蓝 3=绿 4=黄；按 q 退出
  3. 打印 patterns.png（红圆 / 蓝方块 / 绿三角 / 黄圆）举到镜头前即可验证

视频源（三种任一，按你的条件选）
------------------------------
  · 网络视频流：cv2.VideoCapture("http://IP:4747/video")   ← **默认就用这条**
      来源 A：手机装 DroidCam，和电脑连同一 WiFi，app 里看 IP（端口 4747）
              （Android 在 Google Play 搜 DroidCam；**华为**用「出境易」搜 DroidCam 装）
      来源 A2（不用 WiFi）：手机 USB 插电脑 + 开 USB 调试，`adb forward tcp:4747 tcp:4747`，
              然后拉 http://127.0.0.1:4747/video —— 比 WiFi 稳
      来源 B：在宿主机（Windows）跑 `python tools/stream_video.py`，
              它把电脑摄像头推成 http://<宿主机IP>:4747/video，
              虚拟机 / WSL 里直接拉这个 URL（虚拟机没有摄像头时首选这条）
  · USB 摄像头 / 实体摄像头：cv2.VideoCapture(0)
      （虚拟机要用摄像头，需在「虚拟机 → 可移动设备」里直通）
  · 什么都没有：先用 cv2.imread("images/patterns.png") 当一帧，把识别逻辑调对，
      最后再接真实视频源。**线上自学完全允许只用图片/视频文件交差。**

参考思路
--------
  1. 打开视频源：cv2.VideoCapture(URL 或 0)，失败要给人话提示，别直接崩
  2. 循环读帧：while True: ok, frame = cap.read()
     —— 网络流断开时 ok 会变 False，要**自动重连**（release → sleep 0.2 → 重新 open）
  3. 每一帧做「颜色识别 + 质心」（T06 三板斧）：
       转 HSV → 按当前颜色取阈值 → inRange → 形态学开+闭
       → 找最大轮廓 → moments 算质心
       （红色 H 是环形的，要 [0,10] 与 [170,180] 两段合并）
  4. 把质心十字 + 圆 + 坐标文字画到 frame 上
  5. 左上角显示 "Tracking: red" 与 FPS
  6. cv2.imshow + cv2.waitKey(1)：按 q 退出、按 1/2/3/4 切换颜色
  7. 退出前 cap.release() + cv2.destroyAllWindows()

提示（都是踩过坑才总结出来的）
------------------------------
  · 阈值参考（也可以先用 examples/02b_hsv_trackbar.py 拖滑动条自己调）：
        红 H 0–10 与 170–179 | 蓝 H 100–130 | 绿 H 35–85 | 黄 H 20–35
        统一建议 S ≥ 100、V ≥ 60（画面暗就把 V 下限降一点）
  · 按键映射用字典，别用 chr(key) 直接拼字符串：
        KEY = {"1": "red", "2": "blue", "3": "green", "4": "yellow"}
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and chr(key) in KEY: current = KEY[chr(key)]
        if key == ord("q"): break
  · 降延迟：cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        网络流默认缓冲几十帧，画面会越拖越旧；压到 1 帧后延迟从几百 ms 降到几十 ms
  · 中文一律别往画面里写：cv2.putText 不支持中文，会变成一堆问号
        画面文字全用英文（"Tracking: red"），中文留给终端 print
  · 手机/宿主机 IP 存成一个常量（如 DROIDCAM_URL），要改只改一处
  · 处理不过来时丢帧（限流到 ~15 FPS）比"逐帧排队"更实时

自检（本教案不含自动校验脚本，自己定标准自己验）
------------------------------------------
自己写一小段测试：把 images/patterns.png 当"四色帧"，
逐色调用下面这个函数，四种颜色都要能定位、单帧 ≥ 15 FPS：
必须实现下面这个函数（把检测逻辑封装好，后面的程序都靠它）：
    detect_color(frame, color_name) -> (cx, cy) 或 None
    color_name 取 "red" / "blue" / "green" / "yellow"
"""
import os
import sys
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
# from vision_util import setup, imread, FPSMeter, draw_point, put_hud   # noqa: E402

# setup()

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 可改的配置
# 颜色名 -> 它属于第几个数字键（1/2/3/4）
COLOR_KEYS = {"red": "1", "blue": "2", "green": "3", "yellow": "4"}

# 手机 DroidCam 的 IP（换网络就要改），端口固定 4747
DROIDCAM_IP = "192.168.1.100"
DROIDCAM_URL = "http://%s:4747/video" % DROIDCAM_IP

# 没有摄像头时用哪张图/视频练手
FALLBACK_IMAGE = os.path.join(HERE, "images", "patterns.png")

RED_HI1 = (180,255,255)
RED_LO1 = (170,100,50)
RED_HI2 = (10,255,255)
RED_LO2 = (0,100,50)
BLUE_HI = (130,255,255)
BLUE_LO = (100,100,50)
GREEN_HI = (85,255,255)
GREEN_LO = (35,100,50)
YELLOW_HI = (35,255,255)
YELLOW_LO = (26,100,50)
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
# ---------------------------------------------------------------- 必须实现
def detect_color(frame, color_name):
    """在 BGR 帧里找出 color_name 对应颜色目标的质心。

    :param frame: BGR 图（单帧）
    :param color_name: "red" / "blue" / "green" / "yellow"
    :return: (cx, cy) 或 None（没找到）

    提示：转 HSV → inRange（红色两段合并）→ 形态学开+闭 → 最大轮廓 → moments
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: 颜色阈值 -> 形态学 -> 最大轮廓 -> 质心")
    p = []
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = None
    if color_name == "red":
        mask1 = cv2.inRange(frame,RED_LO1,RED_HI1)
        mask2 = cv2.inRange(frame,RED_LO2,RED_HI2)
        mask = cv2.bitwise_or(mask1,mask2)
    elif color_name == "blue":
        mask = cv2.inRange(frame,BLUE_LO,BLUE_HI)
    elif color_name == "green":
        mask = cv2.inRange(frame,GREEN_LO,GREEN_HI)
    elif color_name == "yellow":
        mask = cv2.inRange(frame,YELLOW_LO,YELLOW_HI)
    mask = _morph(mask)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxs=0
    maxcnt = None
    for cnt in cnts:
        # if not _is_circle(cnt):
        #     continue
        if len(cnt[0])>maxs:
            maxs = len(cnt[0])
            maxcnt = cnt
    return _centroid(maxcnt) if maxcnt is not None else None
    # ↑↑↑ 你的代码写在这里 ↑↑↑


def open_source(src):
    """打开视频源并做低延迟设置，返回 cv2.VideoCapture（打不开也要返回对象）。

    src 可能是 0（摄像头）、"http://..."（网络流）或一个视频文件路径。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: VideoCapture + set(CAP_PROP_BUFFERSIZE, 1)")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


def main():
    """主循环：读帧 → 检测当前颜色 → 画 HUD → 按键切换/退出。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # raise NotImplementedError("TODO: 自己写完整程序（含按键切换与断流重连）")
    cap = cv2.VideoCapture(0)          # 0 = 默认摄像头

    if not cap.isOpened():
        print("摄像头打开失败")
        exit()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames = hits = 0
    t0 = time.time()
    while True:
        frames += 1
        ok, frame = cap.read()
        if not ok:
            print("读帧失败")
            break
        result = detect_color(frame,"red")
        if result is not None:
            hits += 1
            x,y=result
            cv2.drawMarker(frame,(int(x),int(y)),(0,255,0),cv2.MARKER_CROSS)
        cv2.imshow("camera", frame)
        if(frames==60):
            print("命中率 %.0f%%   FPS %.1f" % (hits / frames * 100, frames / (time.time() - t0)))
        if cv2.waitKey(1) & 0xFF == ord('q'):   # 按 q 退出
            break

    cap.release()
    cv2.destroyAllWindows()
    # ↑↑↑ 你的代码写在这里 ↑↑↑

if __name__ == "__main__":
    main()
