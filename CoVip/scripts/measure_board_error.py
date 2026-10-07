#!/usr/bin/env python3
"""
Đo sai số đầu bút bằng bảng ArUco — chạy trên LAPTOP, đọc số từ JSON của
node trên Pi (node phải đang chạy với --board).

Mỗi điểm: chạm mũi bút vào đúng điểm trên bảng, giữ yên, bấm Enter. Script
lấy trung bình vài giây rồi in sai số = (số đo) − (số đúng).

    python3 scripts/measure_board_error.py
    python3 scripts/measure_board_error.py --url http://192.168.50.1:8081/ --seconds 4
"""
import argparse
import json
import time
import urllib.request

import numpy as np

# (tên, toạ độ ĐÚNG trong hệ bảng: x phải, y lên, z = cách mặt bảng) — mm
POINTS = [
    ("dấu + ở giữa",          (0, 0, 0)),
    ("góc TRÊN-PHẢI vùng vẽ", (50, 50, 0)),
    ("góc TRÊN-TRÁI",         (-50, 50, 0)),
    ("góc DƯỚI-TRÁI",         (-50, -50, 0)),
    ("góc DƯỚI-PHẢI",         (50, -50, 0)),
]


def collect(url, seconds):
    rows, t_end = [], time.time() + seconds
    while time.time() < t_end:
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=1).read())
        except OSError:
            time.sleep(0.2)
            continue
        if d.get("detected") and d.get("pen_board_x") is not None:
            rows.append([d["pen_board_x"], d["pen_board_y"], d["pen_board_z"],
                         d["x"], d["y"], d["z"],
                         d["board_cam_x"], d["board_cam_y"], d["board_cam_z"],
                         d["board_markers"]])
        time.sleep(0.07)
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.50.1:8081/")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--half-mm", type=float, default=50.0, help="Nửa cạnh vùng vẽ")
    args = ap.parse_args()

    results = []
    for name, true in POINTS:
        true = np.array(true, float) * [args.half_mm / 50, args.half_mm / 50, 1]
        key = input(f"\nChạm mũi bút vào {name}, giữ yên rồi Enter "
                    f"(s = bỏ qua, q = dừng): ").strip().lower()
        if key == "q":
            break
        if key == "s":
            continue
        r = collect(args.url, args.seconds)
        if len(r) < 5:
            print("  ❌ không đủ số liệu (bút hoặc bảng chưa được nhận ra) — bỏ qua điểm này")
            continue
        m, sd = r.mean(axis=0), r.std(axis=0)
        err = m[:3] - true
        results.append(err)
        print(f"  {len(r)} mẫu, {m[9]:.0f} marker")
        print(f"  BANG/CAM (mm)  X {m[6]:+7.1f}  Y {m[7]:+7.1f}  Z {m[8]:7.1f}")
        print(f"  TIP/CAM  (mm)  X {m[3]:+7.1f}  Y {m[4]:+7.1f}  Z {m[5]:7.1f}   "
              f"(rung ±{sd[3]:.1f} ±{sd[4]:.1f} ±{sd[5]:.1f})")
        print(f"  TIP/BANG (mm)  x {m[0]:+7.1f}  y {m[1]:+7.1f}  cách bảng {m[2]:6.1f}")
        print(f"  SAI SỐ   (mm)  x {err[0]:+7.1f}  y {err[1]:+7.1f}  vuông góc bảng {err[2]:+6.1f}"
              f"   → tổng {np.linalg.norm(err):.1f}")

    if results:
        e = np.array(results)
        print(f"\n===== {len(e)} điểm =====")
        print(f"Sai số trung bình (mm): x {e[:,0].mean():+.1f}  y {e[:,1].mean():+.1f}  "
              f"vuông góc bảng {e[:,2].mean():+.1f}")
        print(f"Sai số tổng: trung bình {np.linalg.norm(e, axis=1).mean():.1f} mm, "
              f"lớn nhất {np.linalg.norm(e, axis=1).max():.1f} mm")


if __name__ == "__main__":
    main()
