#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M07 · 摄像头实时检测（已做低延迟优化）
======================================
运行（在 yolo 环境下）：
    cd 07_YOLO推理
    python realtime.py                       # 默认：DroidCam 手机视频流（见下方 DROIDCAM_IP）
    python realtime.py 192.168.1.100         # 换一台手机的 IP（自动补 :4747/video）
    python realtime.py http://x.x.x.x:4747/video   # 任意视频流（含 tools/stream_video.py 的推流）
    python realtime.py usb                   # USB / 实体摄像头 (source=0)
    python realtime.py ../assets/xxx.mp4     # 本地视频文件

手机当摄像头：默认走 DroidCam（大多数人没有额外摄像头，所以默认就用它）
  1. 手机装 DroidCam：Android 在 Google Play 搜 DroidCam；
     **华为**装不了 Google Play 的，用「出境易」这个应用市场搜 DroidCam 安装
     （或用下面的 adb / 实体摄像头方案）
  2. 手机和电脑连**同一个 WiFi**，打开 DroidCam，把它屏幕上显示的 IP 填到下方 DROIDCAM_IP
  3. 端口固定 4747，OpenCV 拉流地址就是 http://<手机IP>:4747/video

其他三种摄像头来源（DroidCam 连不上时按顺序试）：
  · adb（USB 连手机，不依赖 WiFi）：手机开 USB 调试并插上电脑，然后
        adb forward tcp:4747 tcp:4747
        python realtime.py 127.0.0.1      # 走 USB 转发，比 WiFi 稳
  · 实体摄像头（USB）：直接 `python realtime.py usb`；
      虚拟机里要先用「虚拟机 → 可移动设备」把摄像头直通进去
  · 虚拟机完全没有摄像头：在宿主机跑 `python tools/stream_video.py`，
      再在虚拟机里 `python realtime.py http://<宿主机IP>:4747/video`

低延迟三要点（为什么这么做）
----------------------------
1. cap.set(CAP_PROP_BUFFERSIZE, 1)
   网络流默认在 OpenCV 内部缓冲几十帧，画面会越拖越旧。压到 1 帧后，
   读到的总是"最新一帧"，延迟从几百 ms 降到几十 ms。
2. 断流快速重连（sleep 0.2s，而不是长时间阻塞）
   重连期间不卡住，画面恢复更快。
3. 处理不过来就丢帧（MAX_FPS 限流）
   YOLO 单帧推理慢，逐帧排队会越积越多；限流后"看到的就是实时的"。

对照课件：第七章 · 7.4「往届坑」（用 realtime 跑自己的摄像头，交一张带 FPS 的图）
"""
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from vision_util import setup, find_asset                          # noqa: E402

setup()

from ultralytics import YOLO                                       # noqa: E402

DROIDCAM_IP = "192.168.1.100"   # ← 改成你手机 DroidCam 上显示的 IP（默认就用它）
DROIDCAM_PORT = 4747            # 与 DroidCam / tools/stream_video.py 端口一致
DROIDCAM_URL = "http://%s:%d/video" % (DROIDCAM_IP, DROIDCAM_PORT)
MAX_FPS = 15             # 推理帧率上限：处理不过来就丢帧，保证实时
RECONNECT_WAIT = 0.2     # 断流重连间隔（秒），别长时间阻塞


def is_ip_like(s):
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() for p in parts)


def parse_source(argv):
    """参数 → (视频源, 说明文字)。不传参数时先找素材视频，再退回摄像头。"""
    if len(argv) >= 2:
        arg = argv[1]
        if arg.lower() == "usb":
            return 0, "USB 摄像头 (source=0)"
        if arg.startswith("http"):
            return arg, "视频流 " + arg
        if is_ip_like(arg):
            return ("http://%s:%d/video" % (arg, DROIDCAM_PORT),
                    "DroidCam / 推流视频流 (%s)" % arg)
        return arg, "文件 " + arg
    # 默认：DroidCam 手机视频流（最通用——不用额外买摄像头）
    return DROIDCAM_URL, "DroidCam 手机视频流 (%s，改文件顶部 DROIDCAM_IP 可换手机)" % DROIDCAM_IP


def find_weight():
    for cand in ("yolov8n.pt", os.path.join(os.path.expanduser("~"), "yolov8n.pt")):
        if os.path.exists(cand):
            return cand
    return find_asset("yolov8n.pt") or None


def open_capture(source):
    """打开视频源并压低缓冲（降延迟的关键）。"""
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    source, label = parse_source(sys.argv)
    print("使用源：%s" % label)

    weight = find_weight()
    if not weight:
        print("未找到 yolov8n.pt，请先执行： cp ~/yolov8n.pt .")
        sys.exit(1)
    model = YOLO(weight)

    cap = open_capture(source)
    if not cap.isOpened():
        print("无法打开视频源：%s" % source)
        print("  · DroidCam：手机和电脑要在同一 WiFi；确认 app 里的 IP 与 DROIDCAM_IP 一致，端口 4747")
        print("    （华为手机用「出境易」装 DroidCam；也可以用 adb forward tcp:4747 tcp:4747 + 127.0.0.1）")
        print("  · 实体摄像头：python realtime.py usb；虚拟机里要先「虚拟机 → 可移动设备」直通")
        print("  · 虚拟机没摄像头：宿主机跑 python tools/stream_video.py，再拉 http://<宿主机IP>:4747/video")
        sys.exit(1)

    print("开始实时检测（按 q 退出）... 推理帧率上限 %d FPS" % MAX_FPS)
    frame_interval = 1.0 / MAX_FPS
    last_proc = 0.0
    t0, n = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("视频流断开或无新帧，正在重连 ...")
            cap.release()
            time.sleep(RECONNECT_WAIT)
            cap = open_capture(source)
            if not cap.isOpened():
                print("重连失败，退出。请检查视频源连接。")
                break
            continue

        now = time.time()
        if now - last_proc < frame_interval:   # 处理不过来就丢帧
            continue
        last_proc = now
        n += 1

        results = model.predict(source=frame, conf=0.4, verbose=False)
        annotated = results[0].plot()
        fps = n / max(now - t0, 1e-6)
        cv2.putText(annotated, "FPS: %.1f" % fps, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO Realtime", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出。平均 %.1f FPS，共处理 %d 帧（截图可当佐证材料）"
          % (n / max(time.time() - t0, 1e-6), n))


if __name__ == "__main__":
    main()
