#!/usr/bin/env python3
"""
Tìm VÙNG VẼ tốt nhất của tay mới (newarm) trên bảng ArUco thẳng đứng, để thiết kế
kích thước bảng + vị trí đặt bảng.

Bảng thẳng đứng, đối diện "phía trước" tay (-Y), cách trục J1 một đoạn d.
Toạ độ trên bảng: u = ngang (trục X base_link), z = cao (base_link).
Một điểm (u, z) được coi là VẼ ĐƯỢC CHẮC CHẮN nếu ở MỌI độ sâu trong
[d - LIFT - tol, d + tol] (nhấc bút + drone lệch ±tol) đều thoả:
  - có nghiệm IK nhánh vẽ (front=-1, elbow=-1) trong giới hạn khớp (cửa sổ
    servo, đọc từ fk_newarm → theo newarm_servo_calib.json nếu có),
  - khuỷu (J3) và cổ tay (J4) cách mặt bảng >= BODY_MARGIN,
  - góc bút – pháp tuyến bảng <= --max-angle,
  - khuỷu gập >= --min-elbow-bend so với duỗi thẳng (tránh mép tầm với: gần
    điểm kỳ dị, khớp phải quay rất nhanh mới dời được bút → bám kém).
Sau đó tìm hình vuông lớn nhất nằm trọn trong vùng vẽ được, cho từng d.

    python3 newarm_draw_region.py                    # theo cấu hình servo hiện tại
    python3 newarm_draw_region.py --q3-range -30 150 # thử khuỷu lắp lệch
    python3 newarm_draw_region.py --d 0.30 --map     # in bản đồ vùng vẽ ở d=30cm
"""
import argparse
import math

import fk_newarm as F

LIFT_M = 0.02
BODY_MARGIN_M = 0.02
STEP = 0.005
U_RANGE = (-0.20, 0.20)
Z_RANGE = (0.00, 0.50)
BRANCH = (-1, -1)


def _joint_yz(q):
    """Toạ độ y (base_link) của trục J3 và J4 — để kiểm khoảng cách tới bảng."""
    c1, s1 = math.cos(-q[0]), math.sin(-q[0])

    def rot_x(v, a):
        c, s = math.cos(a), math.sin(a)
        return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])

    def world_y(v):
        return F._J1_WORLD[1] + s1 * v[0] + c1 * v[1]

    p2 = F._J2_IN_J1
    a = rot_x(F._J3_IN_J2, -q[1])
    p3 = (p2[0] + a[0], p2[1] + a[1], p2[2] + a[2])
    b = rot_x(F._J4_IN_J3, -q[1] - q[2])
    p4 = (p3[0] + b[0], p3[1] + b[1], p3[2] + b[2])
    return world_y(p3), world_y(p4)


# q3 khi bắp tay - cẳng tay(+bút) thẳng hàng trong mặt phẳng tay
_L2 = F._tip_in_j3(0.0, F.TOOL_OFFSET)
_Q3_STRAIGHT = math.atan2(_L2[2], _L2[1]) - math.atan2(F._J3_IN_J2[2], F._J3_IN_J2[1])


def point_ok(u, z, d, lo, hi, max_angle, min_bend=0.0):
    """Trả góc bút-pháp tuyến (độ) nếu điểm vẽ được ở độ sâu d, ngược lại None."""
    y_plane = F._J1_WORLD[1] - d
    for q, b in F.ik_tip_branches((u, y_plane, z), check_limits=False):
        if b != BRANCH:
            continue
        if not all(l - 1e-9 <= v <= h + 1e-9 for v, l, h in zip(q[:3], lo, hi)):
            return None
        if abs(F._wrap(q[2] - _Q3_STRAIGHT)) < min_bend:
            return None
        y3, y4 = _joint_yz(q)
        if min(y3, y4) - y_plane < BODY_MARGIN_M:
            return None
        v = F.pen_direction(q)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, -v[1]))))
        return ang if ang <= max_angle else None
    return None


def feasible_grid(d, tol, lo, hi, max_angle, min_bend=0.0):
    us = [U_RANGE[0] + i * STEP for i in range(int(round((U_RANGE[1] - U_RANGE[0]) / STEP)) + 1)]
    zs = [Z_RANGE[0] + i * STEP for i in range(int(round((Z_RANGE[1] - Z_RANGE[0]) / STEP)) + 1)]
    depths = [d - LIFT_M - tol, d - LIFT_M, d, d + tol]
    grid = []
    for z in zs:
        row = []
        for u in us:
            worst = 0.0
            for dd in depths:
                a = point_ok(u, z, dd, lo, hi, max_angle, min_bend)
                if a is None:
                    worst = None
                    break
                worst = max(worst, a)
            row.append(worst)
        grid.append(row)
    return us, zs, grid


def largest_square(grid):
    """(side_cells, i_row_bottom, j_col_left) của hình vuông lớn nhất toàn ô vẽ được."""
    n, m = len(grid), len(grid[0])
    dp = [[0] * m for _ in range(n)]
    best = (0, 0, 0)
    for i in range(n):
        for j in range(m):
            if grid[i][j] is None:
                continue
            dp[i][j] = 1 if i == 0 or j == 0 else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            if dp[i][j] > best[0]:
                best = (dp[i][j], i - dp[i][j] + 1, j - dp[i][j] + 1)
    return best


def best_window(grid, cells):
    """Vị trí cửa sổ cells x cells toàn ô vẽ được có góc bút XẤU NHẤT nhỏ nhất.
    Trả (max_ang, mean_ang, i0, j0) hoặc None."""
    n, m = len(grid), len(grid[0])
    best = None
    for i0 in range(n - cells + 1):
        for j0 in range(m - cells + 1):
            mx, sm, ok = 0.0, 0.0, True
            for i in range(i0, i0 + cells):
                row = grid[i]
                for j in range(j0, j0 + cells):
                    a = row[j]
                    if a is None:
                        ok = False
                        break
                    sm += a
                    if a > mx:
                        mx = a
                if not ok:
                    break
            if ok and (best is None or mx < best[0]):
                best = (mx, sm / (cells * cells), i0, j0)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.02, help="dung sai độ sâu ±(m) do drone lệch")
    ap.add_argument("--max-angle", type=float, default=45.0)
    ap.add_argument("--min-elbow-bend", type=float, default=25.0,
                    help="khuỷu phải gập ít nhất ngần này độ so với duỗi thẳng")
    ap.add_argument("--d", type=float, default=None, help="chỉ xét 1 khoảng cách bảng (m)")
    ap.add_argument("--map", action="store_true", help="in bản đồ vùng vẽ được")
    ap.add_argument("--square", type=float, default=None,
                    help="cạnh hình vẽ (m): tìm chỗ đặt hình này có góc bút tốt nhất cho từng d")
    for n in ("q1", "q2", "q3"):
        ap.add_argument(f"--{n}-range", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    args = ap.parse_args()

    lo, hi = list(F.JOINT_LIMITS_LOW[:3]), list(F.JOINT_LIMITS_HIGH[:3])
    for i, n in enumerate(("q1", "q2", "q3")):
        r = getattr(args, f"{n}_range")
        if r:
            lo[i], hi[i] = math.radians(r[0]), math.radians(r[1])
    print("giới hạn q1..q3 (độ): " + ", ".join(f"[{math.degrees(a):.0f},{math.degrees(b):.0f}]" for a, b in zip(lo, hi))
          + f" | dung sai sâu ±{args.tol*100:.0f}cm + nhấc bút {LIFT_M*100:.0f}cm | góc bút <= {args.max_angle:.0f}° | khuỷu gập >= {args.min_elbow_bend:.0f}°")
    print(f"trục J1 ở z={F._J1_WORLD[2]:.3f}m, x={F._J1_WORLD[0]*1000:.1f}mm (base_link)\n")

    ds = [args.d] if args.d else [0.15 + 0.0125 * i for i in range(21)]
    if args.square:
        cells = int(round(args.square / STEP)) + 1
        print(f" d(cm) | chỗ đặt hình {args.square*100:.0f}cm tốt nhất: tâm u (mm) | tâm z (cm) | dưới J1 (cm) | góc bút TB/max (°) | dư mỗi phía (cm)")
        out = []
        for d in ds:
            us, zs, grid = feasible_grid(d, args.tol, lo, hi, args.max_angle, math.radians(args.min_elbow_bend))
            w = best_window(grid, cells)
            if w is None:
                print(f" {d*100:5.1f} |   --")
                continue
            mx, mean, i0, j0 = w
            uc, zc = us[j0] + args.square / 2, zs[i0] + args.square / 2
            # dư: nới cửa sổ ra đều 4 phía tới khi chạm ô không vẽ được
            k = 0
            while (i0 - k - 1 >= 0 and j0 - k - 1 >= 0 and i0 + cells + k < len(zs) and j0 + cells + k < len(us)
                   and all(grid[i][j] is not None
                           for i in range(i0 - k - 1, i0 + cells + k + 1)
                           for j in (j0 - k - 1, j0 + cells + k))
                   and all(grid[i][j] is not None
                           for j in range(j0 - k - 1, j0 + cells + k + 1)
                           for i in (i0 - k - 1, i0 + cells + k))):
                k += 1
            print(f" {d*100:5.1f} | {uc*1000:+34.0f} | {zc*100:10.1f} | {(F._J1_WORLD[2]-zc)*100:12.1f} | "
                  f"{mean:8.1f} / {mx:4.1f}   | {k*STEP*100:6.1f}")
            out.append((mx, d, uc, zc, k * STEP))
        if out:
            mx, d, uc, zc, slack = min(out)
            print(f"\nGÓC BÚT TỐT NHẤT: d={d*100:.1f}cm, tâm u={uc*1000:+.0f}mm, z={zc*100:.1f}cm "
                  f"({(F._J1_WORLD[2]-zc)*100:.1f}cm dưới trục J1), góc max {mx:.1f}°, dư {slack*100:.1f}cm mỗi phía")
        return
    print(" d(cm) | vuông lớn nhất (cm) | tâm u (mm) | tâm z (cm) | dưới trục J1 (cm) | góc bút TB/max trong ô (°)")
    best = None
    for d in ds:
        us, zs, grid = feasible_grid(d, args.tol, lo, hi, args.max_angle, math.radians(args.min_elbow_bend))
        side, i0, j0 = largest_square(grid)
        if side < 2:
            print(f" {d*100:5.1f} |   --")
            continue
        s_m = (side - 1) * STEP
        uc = (us[j0] + us[j0 + side - 1]) / 2
        zc = (zs[i0] + zs[i0 + side - 1]) / 2
        angs = [grid[i][j] for i in range(i0, i0 + side) for j in range(j0, j0 + side)]
        print(f" {d*100:5.1f} | {s_m*100:10.1f}          | {uc*1000:+8.0f}   | {zc*100:8.1f}   | {(F._J1_WORLD[2]-zc)*100:10.1f}        | "
              f"{sum(angs)/len(angs):5.1f} / {max(angs):4.1f}")
        if best is None or s_m > best[0] + 1e-9:
            best = (s_m, d, uc, zc, us, zs, grid, i0, j0, side)
    if best is None:
        print("\nKhông có vùng vẽ nào.")
        return
    s_m, d, uc, zc, us, zs, grid, i0, j0, side = best
    print(f"\nTỐT NHẤT: bảng cách trục J1 {d*100:.1f}cm → vuông {s_m*100:.1f}cm, tâm tại u={uc*1000:+.0f}mm, "
          f"z={zc*100:.1f}cm ({(F._J1_WORLD[2]-zc)*100:.1f}cm dưới trục J1)")
    if args.map or args.d:
        print("\nBản đồ (mỗi ô 1cm; '#' = trong hình vuông tốt nhất, 'o' = vẽ được, '.' = không):")
        for i in range(len(zs) - 1, -1, -2):
            line = "".join(
                "#" if (i0 <= i < i0 + side and j0 <= j < j0 + side) else ("o" if grid[i][j] is not None else ".")
                for j in range(0, len(us), 2))
            if "o" in line or "#" in line:
                print(f" z={zs[i]*100:5.1f} {line}")
        print(f"          u: {us[0]*100:+.0f}cm … {us[-1]*100:+.0f}cm")


if __name__ == "__main__":
    main()
