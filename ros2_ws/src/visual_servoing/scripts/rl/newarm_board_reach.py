#!/usr/bin/env python3
"""
Kiểm tra tay mới (newarm) có vẽ được lên bảng ArUco THẲNG ĐỨNG không, và
bảng phải đặt ở đâu so với tay.

Giả định giống pipeline visual_servoing hiện tại:
  - Bảng thẳng đứng (BoardTransform.R_ideal), vẽ hình vuông SHAPE_SIZE
    (drawing_config.py), nhấc bút lùi LIFT_M theo pháp tuyến giữa các nét
    (robot_config.yaml lift_height_cm).
  - Tay newarm "phía trước" là -Y (q2>0 vung tay về -Y) -> bảng đặt tại
    y = J1_y - d, pháp tuyến hướng vào bảng n = (0,-1,0).
  - Bút gắn cứng, đồng trục J4 (thiết kế cuối) -> hướng bút = hướng cẳng
    tay (q2+q3), IK 3 DOF vị trí quyết định luôn hướng bút.

Với mỗi (d, z_c) = (khoảng cách mặt bảng tới trục J1, độ cao tâm hình trong
base_link): chọn 1 nhánh IK DÙNG CHUNG cho mọi điểm (không nhảy nhánh giữa
nét), yêu cầu: với tới được + trong giới hạn khớp + khuỷu/cổ tay không đâm
xuyên bảng. In góc nghiêng bút so với pháp tuyến bảng xấu nhất trên hình.

    python3 newarm_board_reach.py                      # quét lưới
    python3 newarm_board_reach.py --q3-range -90 90    # khuỷu lắp home giữa
"""
import argparse
import math

import numpy as np

import fk_newarm as F

SHAPE_SIZE = 0.10
POINTS_PER_EDGE = 6
LIFT_M = 0.02
BODY_MARGIN_M = 0.02   # khuỷu (J3) và cổ tay (J4) phải cách mặt bảng ít nhất ngần này
# Giới hạn khớp dùng khi quét (rad). Mặc định: q1, q2 ±90 (servo 180° home
# giữa; cơ khí vai ±118°), q3 [-30,150] (cửa sổ servo 180° khuyến nghị trong
# dải cơ khí khuỷu -110..160, quét va chạm STL). Đổi bằng --q2-range/--q3-range.
LIMITS_LOW = [math.radians(-90), math.radians(-90), math.radians(-30)]
LIMITS_HIGH = [math.radians(90), math.radians(90), math.radians(150)]


def square_board(size, ppe):
    h = size / 2
    corners = [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]
    pts = []
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        for i in range(ppe):
            t = i / ppe
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return pts


def joint_points(q):
    """Vị trí trục J3 (khuỷu) và J4 (cổ tay) trong base_link."""
    def rz(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rx(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    R1 = rz(-q[0])
    R2 = R1 @ rx(-q[1])
    R3 = R2 @ rx(-q[2])
    p2 = np.array(F._J1_WORLD) + R1 @ np.array(F._J2_IN_J1)
    p3 = p2 + R2 @ np.array(F._J3_IN_J2)
    p4 = p3 + R3 @ np.array(F._J4_IN_J3)
    return p3, p4


def evaluate(d, z_c, tool, x_c=0.0):
    """Nhánh IK (front, elbow) giữ nguyên suốt hình."""
    y_plane = F._J1_WORLD[1] - d
    n = np.array([0.0, -1.0, 0.0])
    pts2d = square_board(SHAPE_SIZE, POINTS_PER_EDGE)
    # board-local (u ngang, v đứng) -> base_link; lift lùi về phía tay (+Y)
    targets = []
    for u, v in pts2d:
        targets.append(((x_c + u, y_plane, z_c + v), 0.0))
        targets.append(((x_c + u, y_plane + LIFT_M, z_c + v), LIFT_M))

    best = None
    per_branch = {}
    for (p, lift) in targets:
        best_here = {}
        for q4 in (0.0,):
            for q, b in F.ik_tip_branches(p, q4=q4, tool_offset=tool, check_limits=False):
                if not all(lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in
                           zip(q[:3], LIMITS_LOW, LIMITS_HIGH)):
                    continue
                p3, p4 = joint_points(q)
                if min(y_plane - p3[1], y_plane - p4[1]) > -BODY_MARGIN_M:
                    continue  # khuỷu/cổ tay chạm hoặc xuyên bảng
                v = np.array(F.pen_direction(q))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ n)))))
                margin = min(min(qi - lo, hi - qi) for qi, lo, hi in
                             zip(q[:3], LIMITS_LOW, LIMITS_HIGH))
                key = b
                row = (ang, math.degrees(margin), q)
                if key not in best_here or row[0] < best_here[key][0]:
                    best_here[key] = row
        for key, row in best_here.items():
            per_branch.setdefault(key, []).append(row)
    for b, rows in per_branch.items():
        if len(rows) != len(targets):
            continue  # nhánh này không phủ hết hình
        worst_ang = max(r[0] for r in rows)
        min_margin = min(r[1] for r in rows)
        cand = (worst_ang, -min_margin, b, rows)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool-offset", type=float, nargs=3, default=list(F.TOOL_OFFSET),
                    help="vector hopbut_1 -> đầu bút (m)")
    ap.add_argument("--max-angle", type=float, default=45.0,
                    help="góc bút-pháp tuyến tối đa chấp nhận (tay cũ vẽ ở tilt -35°)")
    for n in ("q1", "q2", "q3"):
        ap.add_argument(f"--{n}-range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                        help=f"giới hạn {n} (độ, URDF) thay cho ±90° — vd lắp lệch home servo")
    ap.add_argument("--quiet", action="store_true", help="chỉ in tóm tắt, không in lưới")
    args = ap.parse_args()
    tool = tuple(args.tool_offset)
    for i, n in enumerate(("q1", "q2", "q3")):
        rng = getattr(args, f"{n}_range")
        if rng is not None:
            LIMITS_LOW[i], LIMITS_HIGH[i] = math.radians(rng[0]), math.radians(rng[1])

    ds = [round(x, 3) for x in np.arange(0.10, 0.46, 0.025)]
    zs = [round(z, 3) for z in np.arange(0.50, -0.06, -0.05)]
    print("giới hạn q1..q3 (độ): " + ", ".join(
        f"[{math.degrees(lo):.0f},{math.degrees(hi):.0f}]" for lo, hi in zip(LIMITS_LOW, LIMITS_HIGH)))
    print(f"TOOL_OFFSET={tool}, hình vuông {SHAPE_SIZE*100:.0f}cm + nhấc bút {LIFT_M*100:.0f}cm, "
          f"J1 tại z={F._J1_WORLD[2]:.3f}")
    print("Ô = góc bút lệch pháp tuyến bảng XẤU NHẤT trên hình (độ); '--' = không vẽ được\n")
    if not args.quiet:
        print("z_c\\d  " + "".join(f"{d*100:6.1f}" for d in ds) + "   (cm)")
    results = []
    for z in zs:
        row = f"{z*100:5.1f}  "
        for d in ds:
            r = evaluate(d, z, tool)
            if r is None:
                row += "    --"
            else:
                row += f"{r[0]:6.0f}"
                results.append((r[0], r[1], d, z, r))
        if not args.quiet:
            print(row)

    ok = [r for r in results if r[0] <= args.max_angle]
    print(f"\nSố vị trí vẽ được: {len(results)}; trong đó góc <= {args.max_angle:.0f}°: {len(ok)}")
    if results:
        ang, neg_margin, d, z, r = min(results, key=lambda t: (t[0], t[1]))
        print(f"TỐT NHẤT: bảng cách trục J1 d={d*100:.1f}cm, tâm hình z={z*100:.1f}cm "
              f"({(F._J1_WORLD[2]-z)*100:.1f}cm dưới trục J1): góc xấu nhất {ang:.1f}°, "
              f"dư giới hạn khớp nhỏ nhất {-neg_margin:.1f}°, nhánh (front,elbow)={r[2]}")
        qs = np.degrees(np.array([row[2] for row in r[3]]))
        print("  dải góc khớp trên hình (URDF, độ): " + " ".join(
            f"q{i+1} [{qs[:, i].min():.0f},{qs[:, i].max():.0f}]" for i in (0, 1, 2)))


if __name__ == "__main__":
    main()
