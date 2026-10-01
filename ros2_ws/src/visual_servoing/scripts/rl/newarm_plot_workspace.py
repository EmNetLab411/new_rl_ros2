#!/usr/bin/env python3
"""
Vẽ phân vùng làm việc của tay mới (newarm) trên bảng ArUco thẳng đứng — để
quan sát bằng mắt kết quả của newarm_draw_region.py.

Mỗi hàng = 1 cách lắp khuỷu. Cột trái: nhìn THẲNG vào mặt bảng (ở khoảng
cách bảng tốt nhất của cách lắp đó). Cột phải: nhìn NGANG (mặt cắt dọc qua
trục J1) — cho thấy bảng được phép lệch gần/xa bao nhiêu.

    python3 newarm_plot_workspace.py                 # -> CoVip/reports/newarm_workspace.png
    python3 newarm_plot_workspace.py --out /tmp/a.png
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import fk_newarm as F
import newarm_draw_region as R

# ── vai trò màu (nền sáng) ──
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REACH_ONLY = "#dddcd6"        # với tới được nhưng không đạt điều kiện vẽ chắc chắn
ACCENT = "#eb6834"            # vùng vẽ 100mm của bảng
SEQ = LinearSegmentedColormap.from_list("pen_angle", ["#d6e6fa", "#2a78d6", "#0c2f66"])

MAX_ANGLE = 45.0
MIN_BEND = math.radians(25.0)
TOL = 0.02
DRAW_MM, MARKER_MM, OFFSET_MM, BOARD_MM = 100.0, 30.0, 75.0, 190.0

CASES = [
    # (nhãn, dải q3 độ, d bảng tốt nhất (m), tâm vùng vẽ (u, z) m) — số từ newarm_draw_region.py --square 0.10
    ("Khuỷu lắp ±90° (sừng servo gắn giữa)", (-90, 90), 0.3125, (-0.020, 0.300)),
    ("Khuỷu lắp lệch −30°…150° (khuyến nghị)", (-30, 150), 0.225, (-0.015, 0.255)),
]


def limits(q3_range):
    lo, hi = list(F.JOINT_LIMITS_LOW[:3]), list(F.JOINT_LIMITS_HIGH[:3])
    lo[2], hi[2] = math.radians(q3_range[0]), math.radians(q3_range[1])
    return lo, hi


def reachable_any(u, z, d):
    """Có nghiệm IK nào (bỏ mọi điều kiện) cho điểm này không."""
    return bool(F.ik_tip_branches((u, F._J1_WORLD[1] - d, z), check_limits=False))


def robust_angle(u, z, d, lo, hi):
    worst = 0.0
    for dd in (d - R.LIFT_M - TOL, d - R.LIFT_M, d, d + TOL):
        a = R.point_ok(u, z, dd, lo, hi, MAX_ANGLE, MIN_BEND)
        if a is None:
            return None
        worst = max(worst, a)
    return worst


def front_grid(d, lo, hi, us, zs):
    ang = np.full((len(zs), len(us)), np.nan)
    reach = np.zeros((len(zs), len(us)), bool)
    for i, z in enumerate(zs):
        for j, u in enumerate(us):
            a = robust_angle(u, z, d, lo, hi)
            if a is not None:
                ang[i, j] = a
            reach[i, j] = reachable_any(u, z, d)
    return ang, reach


def side_grid(u, lo, hi, ds, zs):
    ang = np.full((len(zs), len(ds)), np.nan)
    reach = np.zeros((len(zs), len(ds)), bool)
    for i, z in enumerate(zs):
        for j, d in enumerate(ds):
            a = robust_angle(u, z, d, lo, hi)
            if a is not None:
                ang[i, j] = a
            reach[i, j] = reachable_any(u, z, d)
    return ang, reach


def arm_points(q):
    """(khoảng cách ngang từ trục J1 về phía bảng, z) của J1, J2, J3, J4, đầu bút (m)."""
    def rx(v, a):
        c, s = math.cos(a), math.sin(a)
        return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])
    p1 = (0.0, 0.0, 0.0)
    p2 = F._J2_IN_J1
    a = rx(F._J3_IN_J2, -q[1])
    p3 = tuple(p2[k] + a[k] for k in range(3))
    b = rx(F._J4_IN_J3, -q[1] - q[2])
    p4 = tuple(p3[k] + b[k] for k in range(3))
    t = rx(F._tip_in_j3(0.0, F.TOOL_OFFSET), -q[1] - q[2])
    pt = tuple(p3[k] + t[k] for k in range(3))
    return [(-p[1], F._J1_WORLD[2] + p[2]) for p in (p1, p2, p3, p4, pt)]


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "..", "..", "..", "CoVip", "reports", "newarm_workspace.png")
    ap.add_argument("--out", default=os.path.abspath(default_out))
    ap.add_argument("--step", type=float, default=0.005)
    args = ap.parse_args()

    st = args.step
    us = np.arange(-0.20, 0.20 + 1e-9, st)
    zs = np.arange(0.00, 0.47 + 1e-9, st)
    ds = np.arange(0.08, 0.42 + 1e-9, st)
    j1z = F._J1_WORLD[2] * 100
    j1x = F._J1_WORLD[0] * 100

    X_SIDE_MIN = -16.0     # cm — khuỷu có thể gập ra SAU trục J1
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [40.5, 42.5 - X_SIDE_MIN]})
    im = None
    for row, (label, q3r, d, (uc, zc)) in enumerate(CASES):
        lo, hi = limits(q3r)

        # ── cột trái: nhìn thẳng vào bảng ──
        ax = axes[row][0]
        ang, reach = front_grid(d, lo, hi, us, zs)
        ext = [(us[0] - st / 2) * 100, (us[-1] + st / 2) * 100, (zs[0] - st / 2) * 100, (zs[-1] + st / 2) * 100]
        ax.imshow(np.where(reach, 1.0, np.nan), origin="lower", extent=ext,
                  cmap=LinearSegmentedColormap.from_list("g", [REACH_ONLY, REACH_ONLY]), interpolation="nearest")
        im = ax.imshow(ang, origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=MAX_ANGLE, interpolation="nearest")
        # bảng in: đường cắt, marker, vùng vẽ
        b, m, o, dr = BOARD_MM / 20, MARKER_MM / 10, OFFSET_MM / 10, DRAW_MM / 20
        cx, cz = uc * 100, zc * 100
        ax.add_patch(Rectangle((cx - b, cz - b), 2 * b, 2 * b, fill=False, edgecolor=INK_2, linewidth=1.2, linestyle=(0, (4, 3))))
        for sx in (-1, 1):
            for sz in (-1, 1):
                ax.add_patch(Rectangle((cx + sx * o - m / 2, cz + sz * o - m / 2), m, m,
                                       facecolor=INK, edgecolor=SURFACE, linewidth=1.0))
        ax.add_patch(Rectangle((cx - dr, cz - dr), 2 * dr, 2 * dr, fill=False, edgecolor=ACCENT, linewidth=2.2))
        ax.plot([j1x], [j1z], marker="v", markersize=9, color=INK, markeredgecolor=SURFACE, clip_on=False, zorder=5)
        ax.annotate("trục J1", (j1x, j1z), xytext=(6, 2), textcoords="offset points", color=INK, fontsize=9, va="center")
        ax.annotate("vùng vẽ 10 cm", (cx + dr, cz + dr), xytext=(4, 5), textcoords="offset points",
                    color=INK, fontsize=9, ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85))
        ax.annotate("bảng in 19 cm (4 marker)", (cx - b, cz - b), xytext=(0, -5), textcoords="offset points",
                    color=INK, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85))
        style(ax, "u = trục X base_link (cm) — đứng ở phía tay nhìn vào bảng", "độ cao z trong base_link (cm)")
        ax.set_xlim(ext[1], ext[0]); ax.set_ylim(ext[2], ext[3])   # +X base_link nằm bên TRÁI khi đứng ở phía tay nhìn vào bảng
        ax.set_title(f"{label}\nNhìn thẳng vào bảng — bảng cách trục J1 {d*100:.1f} cm",
                     color=INK, fontsize=11.5, loc="left", pad=10)

        # ── cột phải: nhìn ngang ──
        ax = axes[row][1]
        ang_s, reach_s = side_grid(uc, lo, hi, ds, zs)
        ext2 = [(ds[0] - st / 2) * 100, (ds[-1] + st / 2) * 100, ext[2], ext[3]]
        ax.imshow(np.where(reach_s, 1.0, np.nan), origin="lower", extent=ext2,
                  cmap=LinearSegmentedColormap.from_list("g", [REACH_ONLY, REACH_ONLY]), interpolation="nearest")
        ax.imshow(ang_s, origin="lower", extent=ext2, cmap=SEQ, vmin=0, vmax=MAX_ANGLE, interpolation="nearest")
        ax.plot([d * 100, d * 100], [cz - b, cz + b], color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.plot([d * 100, d * 100], [cz - dr, cz + dr], color=ACCENT, linewidth=3.2, solid_capstyle="butt")
        sols = [q for q, br in F.ik_tip_branches((uc, F._J1_WORLD[1] - d, zc), check_limits=False) if br == R.BRANCH]
        if sols:
            pts = arm_points(sols[0])
            xs, zz = [p[0] * 100 for p in pts], [p[1] * 100 for p in pts]
            ax.plot(xs, zz, color=INK, linewidth=2.0, solid_capstyle="round", zorder=4)
            ax.plot(xs[:-1], zz[:-1], linestyle="none", marker="o", markersize=8, color=INK,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            for name, k, off in (("vai (J2)", 1, (9, 4)), ("khuỷu (J3)", 2, (0, -15)), ("J4", 3, (0, -15)), ("đầu bút", 4, (-5, 9))):
                ax.annotate(name, (xs[k], zz[k]), xytext=off, textcoords="offset points", color=INK, fontsize=9,
                            ha="right" if off[0] < 0 else ("center" if off[0] == 0 else "left"),
                            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.8))
        ax.annotate("mặt bảng", (d * 100, cz - b), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=9,
                    va="bottom", bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.8))
        ax.axvline(0, color=INK_2, linewidth=0.9, linestyle=(0, (1, 3)))
        ax.annotate("trục J1", (0, ext[2]), xytext=(4, 5), textcoords="offset points", color=INK_2, fontsize=9)
        style(ax, "khoảng cách ngang từ trục J1 tới mặt bảng d (cm)", "độ cao z (cm)")
        ax.set_xlim(X_SIDE_MIN, ext2[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title("Nhìn ngang — đầu bút với tới đâu khi bảng gần/xa hơn",
                     color=INK, fontsize=11.5, loc="left", pad=10)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.875, bottom=0.135, hspace=0.27, wspace=0.1)
    cax = fig.add_axes([0.06, 0.058, 0.36, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=9)
    cb.set_label("VẼ ĐƯỢC CHẮC CHẮN — màu = góc bút lệch khỏi pháp tuyến bảng (°), nhạt là tốt", color=INK, fontsize=9.5)
    lx = 0.5
    fig.patches.append(Rectangle((lx, 0.056), 0.018, 0.018, transform=fig.transFigure, facecolor=REACH_ONLY, edgecolor="none"))
    fig.text(lx + 0.024, 0.065, "với tới được nhưng KHÔNG đạt (vượt giới hạn khớp, bút nghiêng > 45°,\n"
             "sát mép tầm với, tay chạm bảng, hoặc hỏng khi bảng lệch ±2 cm / nhấc bút 2 cm)",
             color=INK_2, fontsize=9, va="center")
    fig.patches.append(Rectangle((lx, 0.022), 0.018, 0.018, transform=fig.transFigure, facecolor=SURFACE, edgecolor=GRID))
    fig.text(lx + 0.024, 0.031, "ngoài tầm với", color=INK_2, fontsize=9, va="center")
    fig.text(0.06, 0.962, "Phân vùng làm việc của tay mới trên bảng vẽ thẳng đứng", color=INK, fontsize=16, weight="bold")
    fig.text(0.06, 0.94, f"Tính từ FK/IK (fk_newarm.py), bút dài {abs(F.TOOL_OFFSET[2])*1000:.1f} mm theo CAD. "
             "Khung cam = vùng vẽ 10×10 cm, ô đen = 4 marker ArUco, nét đứt = mép bảng in.",
             color=INK_2, fontsize=10)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"đã ghi {args.out}")


if __name__ == "__main__":
    main()
