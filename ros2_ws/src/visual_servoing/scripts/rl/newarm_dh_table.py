#!/usr/bin/env python3
"""
In bảng Standard DH của tay newarm (J1-J3 + đầu bút) + bộ số kiểm tra để
đối chiếu với công cụ ngoài (vd robotkinematics.study). Tự kiểm DH khớp
fk_newarm.fk_tip() trên 10000 tư thế. Đo lại TOOL_OFFSET thì chạy lại
(a3 và offset θ3 đổi theo).
    python3 newarm_dh_table.py
"""
import sys, math, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import fk_newarm as F


def dh(th, d, a, al):
    ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])


J1 = np.array(F._J1_WORLD); l1 = np.array(F._J3_IN_J2); l2 = np.array(F._tip_in_j3(0.0, F.TOOL_OFFSET))
d1 = J1[2] + F._J2_IN_J1[2]; lat = F._J2_IN_J1[0] + l1[0] + l2[0]
a2 = math.hypot(l1[1], l1[2]); a3 = math.hypot(l2[1], l2[2])
c1 = math.pi/2; c2 = math.atan2(l1[2], l1[1]); c3 = math.atan2(l2[2], l2[1]) - c2
base = np.array([J1[0], J1[1], 0.0])
ROWS = [(c1, d1, 0.0, math.pi/2), (c2, lat, a2, 0.0), (c3, 0.0, a3, 0.0)]


def fk_dh(q):
    T = np.eye(4)
    for qi, (c, d, a, al) in zip(q, ROWS):
        T = T @ dh(-qi + c, d, a, al)
    return T


rng = np.random.default_rng(0); e = 0
for _ in range(10000):
    q = rng.uniform(-1.57, 1.57, 3)
    e = max(e, np.linalg.norm(fk_dh(q)[:3, 3] + base - np.array(F.fk_tip(list(q) + [0.0]))))
print(f"DH vs fk_newarm, 10000 tư thế: sai số max {e*1000:.1e} mm")
angs = []
for _ in range(2000):
    q = rng.uniform(-1.57, 1.57, 3); x3 = fk_dh(q)[:3, 0]; v = np.array(F.pen_direction(list(q) + [0.0]))
    angs.append(math.degrees(math.acos(np.clip(x3 @ v, -1, 1))))
print(f"góc giữa trục x3 (DH) và hướng bút: {min(angs):.2f}..{max(angs):.2f}° (hằng số = không phải lỗi)")
print(f"\nGốc hệ DH trong base_link: ({base[0]*1000:.3f}, {base[1]*1000:.3f}, 0) mm, z0 hướng LÊN")
print("Khớp | θ (độ)            | d (mm)   | a (mm)   | α (độ)")
for i, (c, d, a, al) in enumerate(ROWS, 1):
    print(f" J{i}  | -q{i} {math.degrees(c):+8.3f}     | {d*1000:8.3f} | {a*1000:8.3f} | {math.degrees(al):+.0f}")
print("\nBỘ KIỂM TRA — nhập θ vào web, EE (mm, hệ DH) phải khớp:")
for qd in [(0, 0, 0), (30, 0, 0), (0, 30, 0), (0, 0, 45), (0, 30, 60), (-20, 45, 45), (15, 60, -30)]:
    q = np.radians(qd); T = fk_dh(q)
    th = [(math.degrees(-qi + c) + 180) % 360 - 180 for qi, (c, _, _, _) in zip(q, ROWS)]
    p = T[:3, 3]*1000
    print(f" q_URDF={str(qd):15s} θ_web=({th[0]:+8.3f},{th[1]:+8.3f},{th[2]:+8.3f})  EE_DH=({p[0]:+8.2f},{p[1]:+8.2f},{p[2]:+8.2f})")
q = np.radians((0, 30, 60)); print("\nR của frame 3 tại q=(0,30,60):\n", np.round(fk_dh(q)[:3, :3], 4))
