# -*- coding: utf-8 -*-
"""
课后题 05（实战 · 实时追踪，全自己写）：
========================================
需求：摄像头实时识别指定颜色图案的中心坐标，画面上实时显示
  · 打印 images/patterns.png（红圆/蓝方块/绿三角/黄圆），举着对摄像头
  · 实时画面里：把目标图案的质心画十字 + 圈，显示实时坐标 (x, y)
  · 按 1/2/3/4 切换追踪颜色（红/蓝/绿/黄），按 q 退出

这一题【全部自己写】——只给你思路，不给你代码框架：

视频源（视频流只有网络流一种，即 URL 方式）：
  · 网络视频流：cv2.VideoCapture("http://IP:4747/video")
    只能走这个 URL 形式，来源二选一：
      - 手机 DroidCam（手机装 DroidCam app，和电脑连同一 WiFi，app 里看 IP）
      - 宿主机 stream_video.py 推流（电脑摄像头推成 http://宿主机IP:4747/video）
  · USB 摄像头（不是视频流，是本机设备）：cv2.VideoCapture(0)
  · 没摄像头：先用 cv2.imread 读 images/patterns.png 当一帧，测识别逻辑

参考思路：
  1. 打开视频源：cv2.VideoCapture(URL 或 0)，失败要有提示
  2. 循环读帧：while True: ret, frame = cap.read()（网络流断开要自动重连）
  3. 每一帧做"颜色识别 + 质心"（参考 quiz_01/03 的做法）：
       · 转 HSV → 按当前颜色取阈值 → inRange
       · 形态学去噪（开+闭）→ 找最大轮廓 → moments 算质心
       · 红色 H 是环形要两段合并
  4. 画质心十字 + 圈 + 坐标文字到 frame 上
  5. 画面左上角要显示"正在识别什么颜色" + FPS（⚠ OpenCV putText 不支持中文会乱码，
     画面文字一律用英文，如 "Tracking: red"）
  6. cv2.imshow 显示；waitKey(1) 按 q 退出、按 1/2/3/4 切换颜色
  7. 记得 cap.release() + destroyAllWindows()

提示：
  · 颜色 HSV 阈值：红 H[0,10]+[170,179]，蓝 H[100,130]，绿 H[35,85]，黄 H[20,35]
  · 按键映射：1→红 2→蓝 3→绿 4→黄（要用 dict 把数字键转成颜色名，别直接用 chr(key) 匹配）
  · 延迟优化：cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 能明显降低网络流延迟
  · 网络摄像头把手机 IP 存成一个常量 DROIDCAM_IP，改一个地方即可
  · 参考课件「实战：实时追踪」页 + 例程 02b/03 + Day2 的 realtime.py
"""
import cv2
import numpy as np

# ↓↓↓ 全部由你实现：打开摄像头 → 循环识别指定颜色质心 → 显示坐标 ↓↓↓
