#!/usr/bin/env python3
"""
Tạo bảng workspace ArUco để IN (A4, tỉ lệ 1:1) cho tay mới (newarm).

Thiết kế theo vùng vẽ tốt nhất tính từ FK/IK (scripts/rl/newarm_draw_region.py):
vùng vẽ 100x100mm ở giữa, 4 marker DICT_4X4_1000 ID 0-3 (0 trên-trái, 1 trên-phải,
2 dưới-phải, 3 dưới-trái) cạnh 30mm, tâm marker cách tâm bảng ±75mm.
Số này phải khớp config/vision_board_newarm.yaml (board_marker_offset_m / _size_m).

    python3 make_workspace_board.py        # -> workspace_board_newarm_A4.pdf + .png
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600
DRAW_MM = 100.0      # vùng vẽ (= SHAPE_SIZE 0.10 trong drawing_config.py)
MARKER_MM = 30.0
OFFSET_MM = 75.0     # tâm marker cách tâm bảng
BOARD_MM = 190.0     # đường cắt: chừa 5mm trắng quanh marker
CENTER_FROM_TOP_MM = 112.0
OUT = "workspace_board_newarm_A4"

mm = DPI / 25.4
W, H = int(round(210 * mm)), int(round(297 * mm))
cx, cy = W / 2, CENTER_FROM_TOP_MM * mm


def P(x, y):
    """mm trong hệ bảng (x phải, y lên, gốc ở tâm) -> pixel."""
    return int(round(cx + x * mm)), int(round(cy - y * mm))


img = np.full((H, W), 255, np.uint8)
adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ms = int(round(MARKER_MM * mm))
POS = {0: (-OFFSET_MM, OFFSET_MM), 1: (OFFSET_MM, OFFSET_MM), 2: (OFFSET_MM, -OFFSET_MM), 3: (-OFFSET_MM, -OFFSET_MM)}
for mid, (x, y) in POS.items():
    x0, y0 = P(x - MARKER_MM / 2, y + MARKER_MM / 2)
    img[y0:y0 + ms, x0:x0 + ms] = cv2.aruco.generateImageMarker(adict, mid, ms)

G, t = 175, max(1, int(round(0.15 * mm)))
cv2.rectangle(img, P(-BOARD_MM / 2, BOARD_MM / 2), P(BOARD_MM / 2, -BOARD_MM / 2), G, t)   # đường cắt
h, L = DRAW_MM / 2, 6.0
for sx in (-1, 1):                                                                        # 4 góc vùng vẽ
    for sy in (-1, 1):
        cv2.line(img, P(sx * h, sy * h), P(sx * (h - L), sy * h), G, t)
        cv2.line(img, P(sx * h, sy * h), P(sx * h, sy * (h - L)), G, t)
cv2.line(img, P(-3, 0), P(3, 0), G, t)                                                    # tâm bảng
cv2.line(img, P(0, -3), P(0, 3), G, t)
yb = -BOARD_MM / 2 - 12                                                                   # thước 100mm
cv2.line(img, P(-50, yb), P(50, yb), 0, 2 * t)
for k in range(0, 101, 10):
    cv2.line(img, P(-50 + k, yb), P(-50 + k, yb + (4 if k % 50 == 0 else 2.5)), 0, 2 * t)

pil = Image.fromarray(img)
dr = ImageDraw.Draw(pil)


def font(pt, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu/", ""):
        try:
            return ImageFont.truetype(d + name, int(round(pt / 72 * DPI)))
        except OSError:
            pass
    return ImageFont.load_default()


def text(s, x, y, pt=8, fill=0, bold=False, anchor="la"):
    dr.text(P(x, y), s, font=font(pt, bold), fill=fill, anchor=anchor)


text("▲ TRÊN", 0, BOARD_MM / 2 + 7, 11, 90, True, "mm")
for mid, (x, y) in POS.items():
    text(f"ID {mid}", x, y + (MARKER_MM / 2 + 2.2) * (1 if y > 0 else -1), 6, 150, anchor="mm")
text("vùng vẽ 100 × 100 mm", 0, -h + 2.5, 6, 175, anchor="mm")
text("100 mm — in xong đo lại bằng thước, phải đúng 100 mm", -50, yb - 2.5, 7.5)
lines = [
    ("Bảng workspace visual servoing — tay mới (newarm)", True),
    ("DICT_4X4_1000, ID 0–3 · marker 30 mm · tâm marker cách tâm bảng ±75 mm · vùng vẽ 100 × 100 mm", False),
    ("In 100 % / Actual size (KHÔNG chọn “Fit to page”). Dán phẳng lên bìa cứng, cắt theo viền xám hoặc rộng hơn.", False),
    ("Đặt bảng thẳng đứng, đối diện phía trước tay; tâm bảng (dấu +) thẳng trục J1 và thấp hơn trục J1 ≈ 160 mm;", False),
    ("mặt bảng cách trục J1 ≈ 225 mm (khuỷu lắp dải −30°…150°; vùng dùng được 170–290 mm).", False),
    ("Chạy detector với config/vision_board_newarm.yaml (offset 0.075, size 0.030).", False),
]
y = yb - 11
for s, b in lines:
    text(s, -BOARD_MM / 2, y, 8.5 if b else 7.5, bold=b)
    y -= 5.2 if b else 4.6

pil.save(OUT + "_600dpi.png", dpi=(DPI, DPI))
pil.save(OUT + ".pdf", "PDF", resolution=DPI)

# ── tự kiểm: detector đọc lại + solvePnP với đúng hình học bảng ──
arr = np.array(pil)
small = cv2.resize(arr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
c, ids, _ = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters()).detectMarkers(small)
got = {int(i): cc[0] * 4 for i, cc in zip(ids.flatten(), c)}
assert sorted(got) == [0, 1, 2, 3], f"detect được {sorted(got)}"
o, hm = OFFSET_MM / 1000, MARKER_MM / 2000
obj, pts = [], []
for mid, (sx, sy) in {0: (-1, 1), 1: (1, 1), 2: (1, -1), 3: (-1, -1)}.items():
    x, y0 = sx * o, sy * o
    obj += [[x - hm, y0 + hm, 0], [x + hm, y0 + hm, 0], [x + hm, y0 - hm, 0], [x - hm, y0 - hm, 0]]
    pts += list(got[mid])
f = 0.5 / 0.0254 * DPI          # "camera" ảo nhìn thẳng tờ giấy từ 0.5m
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
ok, rvec, tvec = cv2.solvePnP(np.array(obj, np.float32), np.array(pts, np.float32), K, None)
print(f"detect đủ ID {sorted(got)}; solvePnP với offset {OFFSET_MM}mm/size {MARKER_MM}mm: "
      f"z = {tvec[2,0]*1000:.1f} mm (đúng = 500.0), tâm lệch ({tvec[0,0]*1000:+.2f}, {tvec[1,0]*1000:+.2f}) mm, "
      f"xoay {np.degrees(np.linalg.norm(rvec)) % 180:.2f}°")
print(f"đã ghi {OUT}.pdf và {OUT}_600dpi.png ({210}x{297} mm @ {DPI} dpi)")
