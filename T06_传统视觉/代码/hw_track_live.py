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

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from vision_util import setup, imread, FPSMeter, draw_point, put_hud   # noqa: E402

setup()

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 可改的配置
# 颜色名 -> 它属于第几个数字键（1/2/3/4）
COLOR_KEYS = {"red": "1", "blue": "2", "green": "3", "yellow": "4"}

# 手机 DroidCam 的 IP（换网络就要改），端口固定 4747
DROIDCAM_IP = "192.168.1.100"
DROIDCAM_URL = "http://%s:4747/video" % DROIDCAM_IP

# 没有摄像头时用哪张图/视频练手
FALLBACK_IMAGE = os.path.join(HERE, "images", "patterns.png")


# ---------------------------------------------------------------- 必须实现
def detect_color(frame, color_name):
    """在 BGR 帧里找出 color_name 对应颜色目标的质心。

    :param frame: BGR 图（单帧）
    :param color_name: "red" / "blue" / "green" / "yellow"
    :return: (cx, cy) 或 None（没找到）

    提示：转 HSV → inRange（红色两段合并）→ 形态学开+闭 → 最大轮廓 → moments
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO: 颜色阈值 -> 形态学 -> 最大轮廓 -> 质心")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


def open_source(src):
    """打开视频源并做低延迟设置，返回 cv2.VideoCapture（打不开也要返回对象）。

    src 可能是 0（摄像头）、"http://..."（网络流）或一个视频文件路径。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO: VideoCapture + set(CAP_PROP_BUFFERSIZE, 1)")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


def main():
    """主循环：读帧 → 检测当前颜色 → 画 HUD → 按键切换/退出。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO: 自己写完整程序（含按键切换与断流重连）")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


if __name__ == "__main__":
    main()
