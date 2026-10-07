"""
Forward/Inverse Kinematics cho cánh tay MỚI — thiết kế CUỐI
ref/newarm_final_description-20261001T060843Z-1-001 (URDF export từ Fusion
360 bằng plugin syuntoku14). Pure Python (không numpy), cùng phong cách
fk_ik_utils.py — file đó giữ nguyên cho tay 6-DOF cũ.

4 servo, KHÔNG có J5/gripper; hộp bút (hopbut_1) + bút (but_1) gắn cứng vào
đầu ra J4:
  J1 base        "Revolute 2"   TD-8120MG  trục (0,0,-1)  yaw
  J2 shoulder    "Revolute 3"   RDS3120    trục (-1,0,0)  pitch
  J3 elbow       "Revolute 7"   MG996R     trục (-1,0,0)  pitch
  J4 wrist_roll  "Revolute 24"  MG996R     trục (0,0,-1)  xoay quanh trục cẳng tay

Bút ĐỒNG TRỤC với J4 (kiểm từ but_1.stl) -> J4 không dời đầu bút, không đổi
hướng bút, chỉ xoay bút/hộp bút quanh trục của nó (dùng để quay marker về
phía camera). Vị trí đầu bút do J1-J3 quyết định; hướng bút = hướng cẳng
tay, nghiêng khỏi phương thẳng đứng đúng q2+q3 — không có wrist pitch.

Toàn bộ rpy trong URDF = 0; ở q=0 tay treo thẳng xuống (-Z) từ servo J1
(z=0.418) tới đầu bút (z=0.013). LƯU Ý: các file STL được export ở tư thế
KHÁC (tay duỗi ngang, q2=+90°) — không dùng trực tiếp toạ độ lưới làm q=0.

SỬA LỖI EXPORT URDF:
  "Revolute 2" có origin (0.413256, 0.037042, -0.014973) -> trục yaw lệch
  0.42m khỏi tay. Trục thật = trục ra của servo 8120 trong
  DigitalServo8120_1.stl: (x=-0.01315, y=0.0250), cắt trục J2. Đã dời origin J1 về đó
  và bù lại origin J2 để mọi frame sau giữ nguyên vị trí ở q=0. J2/J3/J4
  khớp tâm bánh răng STL (sau khi xoay lưới về tư thế thẳng xuống).
"""

import json
import math
import os
from typing import Sequence, Tuple

JOINT_NAMES = ("base", "shoulder", "elbow", "wrist_roll")

# ── Servo ────────────────────────────────────────────────────────────────────
# Driver wicom_roboarm map lệnh 0-180 "độ" TUYẾN TÍNH sang pulse_min..pulse_max
# (_angle_to_pulse_us). Cả 4 servo là bản 180° (khớp URDF limit ±90°) -> 1 độ
# lệnh = 1 độ thật (range_deg=180; servo 270° thì đặt 270).
#   home_deg : lệnh servo ứng với q=0 (tay thẳng xuống)
#   inverted : servo lắp ngược chiều trục URDF
# Giá trị dưới đây là MẶC ĐỊNH; số hiệu chỉnh thật (sau khi lắp) nằm trong
# newarm_servo_calib.json cạnh file này — do CoVip/scripts/newarm_bringup.py
# ghi, tự nạp đè lúc import (xem _load_calib()).
SERVO_SPECS = {
    "base":       {"model": "TD-8120MG", "range_deg": 180.0, "home_deg": 90.0, "inverted": False},
    "shoulder":   {"model": "RDS3120",   "range_deg": 180.0, "home_deg": 90.0, "inverted": False},
    "elbow":      {"model": "MG996R",    "range_deg": 180.0, "home_deg": 90.0, "inverted": False},
    "wrist_roll": {"model": "MG996R",    "range_deg": 180.0, "home_deg": 90.0, "inverted": False},
}


def servo_deg_to_q(joint: str, servo_deg: float) -> float:
    """Độ lệnh servo (thang 0-180 của driver) -> góc khớp URDF (rad)."""
    s = SERVO_SPECS[joint]
    q = math.radians((float(servo_deg) - s["home_deg"]) * s["range_deg"] / 180.0)
    return -q if s["inverted"] else q


def q_to_servo_deg(joint: str, q: float) -> float:
    """Góc khớp URDF (rad) -> độ lệnh servo (thang 0-180 của driver), đã kẹp."""
    s = SERVO_SPECS[joint]
    if s["inverted"]:
        q = -q
    deg = s["home_deg"] + math.degrees(q) * 180.0 / s["range_deg"]
    return max(0.0, min(180.0, deg))


# ── Đầu bút ──────────────────────────────────────────────────────────────────
# Vector từ gốc frame hopbut_1 (hộp bút, trên trục J4, 2mm dưới frame bánh
# răng J4) tới ĐẦU BÚT, trong hệ hopbut_1 (m). Lấy từ but_1.stl: bút đồng
# trục J4 (lệch < 0.1mm), mũi bút thấp hơn gốc hopbut 62.5mm (CAD, lò xo
# không nén). Đo thật có thể khác vài mm (lò xo/cách đo) — đo lại khi lắp.
# Chỉ ảnh hưởng dự đoán ROI (Phase 4) và vẽ; hand-eye (Phase 3) không cần.
TOOL_OFFSET = (0.0, 0.0, -0.0625)

# Biến môi trường NEWARM_CALIB trỏ sang file khác (vd file riêng cho mô phỏng),
# để chạy thử trong Gazebo không đụng tới số hiệu chỉnh của tay thật.
CALIB_PATH = os.environ.get("NEWARM_CALIB") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "newarm_servo_calib.json")


def _load_calib(path=CALIB_PATH):
    """Nạp đè SERVO_SPECS / TOOL_OFFSET từ file hiệu chỉnh nếu có:
    {"servos": {"elbow": {"home_deg": 30.0, "inverted": true}}, "tool_offset": [0,0,-0.058]}"""
    global TOOL_OFFSET
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    for name, over in data.get("servos", {}).items():
        SERVO_SPECS[name].update(over)
    if "tool_offset" in data:
        TOOL_OFFSET = tuple(float(v) for v in data["tool_offset"])
    return True


CALIB_LOADED = _load_calib()


def _servo_window(joint):
    """Dải góc khớp (rad) mà servo với tới được với lệnh 0..180."""
    s = SERVO_SPECS[joint]
    a = math.radians((0.0 - s["home_deg"]) * s["range_deg"] / 180.0)
    b = math.radians((180.0 - s["home_deg"]) * s["range_deg"] / 180.0)
    if s["inverted"]:
        a, b = -b, -a
    return a, b


# Giới hạn khớp (rad, q=0 là tay treo thẳng xuống) = cửa sổ servo 0..180 quanh
# home_deg. Mặc định (home 90) ra đúng ±90° như URDF; lắp lệch sừng khuỷu
# (vd home_deg=30) thì thành [-30°, 150°]. Giới hạn cơ khí (quét va chạm
# STL): vai ±118°, khuỷu -110°..+160° — cửa sổ servo phải nằm trong đó.
JOINT_LIMITS_LOW = tuple(_servo_window(n)[0] for n in JOINT_NAMES)
JOINT_LIMITS_HIGH = tuple(_servo_window(n)[1] for n in JOINT_NAMES)

# ── helpers ──────────────────────────────────────────────────────────────────

def _I():
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def _T(x,y,z):
    return [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]]

def _Rx(t):
    c,s=math.cos(t),math.sin(t)
    return [[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]]

def _Rz(t):
    c,s=math.cos(t),math.sin(t)
    return [[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]

def _mul(A,B):
    R=[[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                R[i][j]+=A[i][k]*B[k][j]
    return R

def _chain(*ms):
    r=_I()
    for m in ms: r=_mul(r,m)
    return r

def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

# ── chuỗi động học (số lấy từ file .xacro của thiết kế) ──────────────────────────

_RIGID1 = (-0.009205, 0.011786, 0.433355)      # base_link -> DigitalServo8120_1
_J1_ORIGIN = (-0.003945, 0.013187, -0.014973)  # SỬA: URDF ghi (0.413256, 0.037042, -0.014973)
_J2_ORIGIN = (0.026245, 0.0, -0.033247)        # SỬA (bù J1): URDF ghi (-0.390956, -0.023855, -0.033247)
_RIGID4 = (-0.01095, 0.00495, -0.027)          # rds3120_support_1 -> link1_1
_RIGID5 = (-0.019829, -0.029968, -0.1)         # -> Servo_mg996_BracketM_1
_RIGID6 = (-0.008, 0.04925, -0.01987)          # -> MG996R_servo_1
_J3_ORIGIN = (-0.0155, -0.03415, 0.005)
_RIGID8 = (-0.002, 0.0, 0.0)                   # banhrang_mg996_1 -> ServoBracketU_1
_RIGID22 = (0.0446, 0.0, -0.05)                # -> link2_1
_RIGID23 = (-0.023585, -0.024161, -0.1)        # -> MG996R_servo__1__1
_J4_ORIGIN = (0.005, 0.01415, -0.0155)
_RIGID25 = (0.0, 0.0, -0.002)                  # banhrang_mg996__1__1 -> hopbut_1

_J1_WORLD = _add(_RIGID1, _J1_ORIGIN)                           # điểm trên trục J1
_J2_IN_J1 = _J2_ORIGIN                                          # trục J2 trong hệ J1
_J3_IN_J2 = _add(_add(_RIGID4, _RIGID5), _add(_RIGID6, _J3_ORIGIN))
_J4_IN_J3 = _add(_add(_RIGID8, _RIGID22), _add(_RIGID23, _J4_ORIGIN))


def fk_flange_matrix(q: Sequence[float]):
    """Ma trận 4x4 của hopbut_1 (hộp bút, gắn cứng sau J4 — nơi dán marker)
    trong base_link. q: 4 góc URDF (rad) [base, shoulder, elbow, wrist_roll]."""
    if len(q) != 4:
        raise ValueError(f"Expected 4 joint angles {JOINT_NAMES}, got {len(q)}")
    return _chain(
        _T(*_J1_WORLD), _Rz(-q[0]),            # axis (0,0,-1)
        _T(*_J2_IN_J1), _Rx(-q[1]),            # axis (-1,0,0)
        _T(*_J3_IN_J2), _Rx(-q[2]),            # axis (-1,0,0)
        _T(*_J4_IN_J3), _Rz(-q[3]),            # axis (0,0,-1)
        _T(*_RIGID25),
    )


def fk_tip_matrix(q: Sequence[float], tool_offset=TOOL_OFFSET):
    """Như fk_flange_matrix() nhưng dời gốc tới đầu bút (TOOL_OFFSET)."""
    return _mul(fk_flange_matrix(q), _T(*tool_offset))


def fk_tip(q: Sequence[float], tool_offset=TOOL_OFFSET) -> Tuple[float, float, float]:
    """Vị trí đầu bút (x, y, z) trong base_link (m)."""
    M = fk_tip_matrix(q, tool_offset)
    return (M[0][3], M[1][3], M[2][3])


def pen_direction(q: Sequence[float]) -> Tuple[float, float, float]:
    """Vector đơn vị hướng bút (trục -Z của hopbut_1) trong base_link.
    Nghiêng khỏi phương thẳng đứng (xuống) đúng q2+q3."""
    M = fk_flange_matrix(q)
    return (-M[0][2], -M[1][2], -M[2][2])


# ── IK giải tích ─────────────────────────────────────────────────────────────
# Với q4 cố định, vector J3 -> đầu bút là HẰNG trong hệ J3, nên bài toán vị trí
# 3 ẩn (q1,q2,q3) tách được:
#   - J2, J3 đều quay quanh trục x của hệ J1 -> thành phần x (lệch ngang)
#     không đổi = X_LAT, (y,z) là tay 2 khâu phẳng.
#   - J1 quay quanh trục đứng qua _J1_WORLD -> tìm q1 sao cho điểm (X_LAT, y)
#     quay tới đúng (dx, dy).
# J4 KHÔNG chỉnh được hướng bút (bút đồng trục J4) — truyền q4 tuỳ ý.

def _tip_in_j3(q4, tool_offset):
    c, s = math.cos(-q4), math.sin(-q4)
    tx, ty, tz = tool_offset[0], tool_offset[1], tool_offset[2] + _RIGID25[2]
    return _add(_J4_IN_J3, (c*tx - s*ty, s*tx + c*ty, tz))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ik_tip(target, q4=0.0, tool_offset=TOOL_OFFSET, check_limits=True):
    """IK vị trí đầu bút. target (x,y,z) base_link (m).
    Trả list nghiệm [q1,q2,q3,q4] (rad, URDF) — tối đa 4 nghiệm (2 hướng
    base x 2 khuỷu), đã lọc giới hạn khớp nếu check_limits. [] nếu không với tới."""
    return [q for q, _ in ik_tip_branches(target, q4, tool_offset, check_limits)]


def ik_tip_branches(target, q4=0.0, tool_offset=TOOL_OFFSET, check_limits=True):
    """Như ik_tip() nhưng trả [(q, (front, elbow))]: front=+1/-1 (đầu bút ở
    phía +/- trục y của hệ J1), elbow=+1/-1. Khi bám 1 nét vẽ liên tục phải
    giữ NGUYÊN 1 nhánh, nhảy nhánh = tay quật mạnh giữa chừng."""
    l2 = _tip_in_j3(q4, tool_offset)
    l1 = _J3_IN_J2
    x_lat = _J2_IN_J1[0] + l1[0] + l2[0]

    dx = target[0] - _J1_WORLD[0]
    dy = target[1] - _J1_WORLD[1]
    dz = target[2] - _J1_WORLD[2]
    r2 = dx*dx + dy*dy - x_lat*x_lat
    if r2 < 0:
        return []
    sols = []
    for front, y1 in ((1, math.sqrt(r2)), (-1, -math.sqrt(r2))):
        # Rz(-q1) @ (x_lat, y1) = (dx, dy)
        q1 = -_wrap(math.atan2(dy, dx) - math.atan2(y1, x_lat))
        # tay 2 khâu trong mặt (y,z) của hệ J1, gốc tại trục J2
        py = y1 - _J2_IN_J1[1]
        pz = dz - _J2_IN_J1[2]
        a = (l1[1], l1[2])
        b = (l2[1], l2[2])
        la, lb = math.hypot(*a), math.hypot(*b)
        d2 = py*py + pz*pz
        cos_g = (d2 - la*la - lb*lb) / (2*la*lb)
        if abs(cos_g) > 1.0:
            continue
        g = math.acos(cos_g)                       # góc giữa a và (Rx b)
        base_ang = math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])
        for elbow, sgn in ((1, 1.0), (-1, -1.0)):
            # Rx(-q3) quay b thêm góc -q3 trong mặt (y,z)
            q3 = _wrap(-(sgn*g - base_ang))
            c3, s3 = math.cos(-q3), math.sin(-q3)
            by, bz = c3*b[0] - s3*b[1], s3*b[0] + c3*b[1]
            vy, vz = a[0] + by, a[1] + bz
            q2 = _wrap(-(math.atan2(pz, py) - math.atan2(vz, vy)))
            q = [q1, q2, q3, q4]
            if check_limits and not all(
                    lo - 1e-9 <= v <= hi + 1e-9
                    for v, lo, hi in zip(q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)):
                continue
            if all(max(abs(_wrap(u - v)) for u, v in zip(q, o)) > 1e-6 for o, _ in sols):
                sols.append((q, (front, elbow)))
    return sols


def ik_tip_nearest(target, q_prev, tool_offset=TOOL_OFFSET):
    """Nghiệm IK gần q_prev nhất (giữ q4 của q_prev) — dùng khi bám liên tục
    để không nhảy nhánh. None nếu không với tới / vượt giới hạn."""
    sols = ik_tip(target, q4=q_prev[3], tool_offset=tool_offset)
    if not sols:
        return None
    return min(sols, key=lambda q: max(abs(_wrap(a - b)) for a, b in zip(q, q_prev)))


def servo_degs_to_q(servo_degs: Sequence[float]):
    """4 độ lệnh servo [base, shoulder, elbow, wrist_roll] -> 4 góc URDF (rad)."""
    return [servo_deg_to_q(n, d) for n, d in zip(JOINT_NAMES, servo_degs)]


def test_fk():
    for label, q in [
        ("home (0,0,0,0)", [0, 0, 0, 0]),
        ("J1=+30°", [0.5236, 0, 0, 0]),
        ("J2=+30°", [0, 0.5236, 0, 0]),
        ("J3=+30°", [0, 0, 0.5236, 0]),
        ("J4=+30°", [0, 0, 0, 0.5236]),
    ]:
        x, y, z = fk_tip(q)
        print(f"{label:16s}: tip x={x:+.4f} y={y:+.4f} z={z:+.4f}")


if __name__ == "__main__":
    test_fk()
