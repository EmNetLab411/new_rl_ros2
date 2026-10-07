#!/usr/bin/env python3
"""
Hand-eye calibration (Phase 3 trong PLAN.md): tìm T_cam_to_base — ma trận
4x4 biến đổi 1 điểm từ hệ camera sang hệ base_link (p_base = T_cam_to_base
@ p_cam_homogeneous).

Bố trí vật lý của bài toán này: CAMERA CỐ ĐỊNH trong hệ base_link (gắn
trên khung drone cùng cánh tay), còn MARKER/board ArUco gắn tạm trên phần
DI CHUYỂN của cánh tay (gần điểm gắn bút) — đây là kiểu "eye-to-hand",
NGƯỢC với mặc định "eye-in-hand" (camera gắn trên phần di chuyển, target
cố định) mà cv2.calibrateHandEye() giả định sẵn. Cách xử lý chuẩn: đưa
NGHỊCH ĐẢO của gripper2base (tức base2gripper) vào hàm thay vì gripper2base
gốc — kết quả trả về khi đó chính là cam2base (= T_cam_to_base) thay vì
cam2gripper. Đây là kỹ thuật chuẩn cho eye-to-hand, không phải suy diễn.

Điểm hay của thuật toán Tsai-Lenz (bên trong cv2.calibrateHandEye): nó
dùng CHUYỂN ĐỘNG TƯƠNG ĐỐI giữa các cặp tư thế để giải, nên KHÔNG cần biết
trước offset thật giữa marker và điểm gắn bút (bibut_1) — chỉ cần offset
đó CỐ ĐỊNH suốt quá trình đo (đúng giả định đã chốt: "Tip và đĩa xanh phải
cố định cứng với nhau"). Phần self-test dưới đây kiểm đúng tính chất này.

Marker dùng: 1 marker ArUco ĐƠN (không phải board nhiều marker) gắn trực
tiếp lên đĩa cứng ngay sát điểm gắn bút — KHÔNG dùng
`vs_lib/vision/vision_aruco_detector.py` có sẵn trong package
visual_servoing vì node đó viết cho board 4 marker (DICT_4X4_1000, cần
thấy đồng thời >=2 marker cùng lúc) — khác dictionary VÀ khác cách bố trí
với 1 marker đơn DICT_4X4_50. Script này tự làm phần detect single-marker
riêng (đọc thẳng ảnh từ topic `/aeroscript/pen_image` + K/dist từ
`calib/c930e_720p.npz` đã có ở Phase 2), không phụ thuộc node/topic nào
khác của package visual_servoing.

CÁCH 2 — KHÔNG DÁN MARKER (`collect-tip`): dùng chính đầu bút làm điểm đo.
Mỗi tư thế ghi 1 cặp điểm: đầu bút theo FK (base_link) và XYZ đầu bút camera
đo (/aeroscript/pen_xyz, mm). Khớp 2 tập điểm (Kabsch) ra T_cam_to_base.
XYZ bút nhiễu hơn marker nên cần >= 12 tư thế trải rộng theo CẢ 3 chiều
(không đồng phẳng). Phụ thuộc TOOL_OFFSET đúng (khác cách marker).

Chạy kiểm logic ngay bây giờ (không cần phần cứng):
    python3 scripts/calibrate_hand_eye.py --self-test

Chạy thu thập dữ liệu thật (cần Phase 2 xong + robot + camera + marker):
    python3 scripts/calibrate_hand_eye.py collect --n-poses 15 \\
        --marker-id 0 --marker-size-mm 20 --dict DICT_4X4_50
    # marker do run_pi4_ros2.py --tool-marker-mm 20 dò sẵn trên ảnh thô:
    python3 scripts/calibrate_hand_eye.py collect --marker-size-mm 20 \\
        --pose-topic /aeroscript/tool_marker_pose

Không dán marker, tay tự đi qua lưới tư thế (cần bút + run_pi4_ros2.py):
    python3 scripts/calibrate_hand_eye.py collect-tip --auto
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "calib"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_models import add_arm_args, arm_from_args  # noqa: E402


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def solve_hand_eye_eye_to_hand(T_gripper2base_list, T_target2cam_list,
                                method=cv2.CALIB_HAND_EYE_TSAI):
    """Eye-to-hand: trả về T_cam_to_base (4x4), sao cho
    p_base_homog = T_cam_to_base @ p_cam_homog."""
    if len(T_gripper2base_list) < 3:
        raise ValueError("Cần ít nhất 3 tư thế, khuyến nghị 10-15 để ổn định.")

    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for T_g2b, T_t2c in zip(T_gripper2base_list, T_target2cam_list):
        T_b2g = invert_T(T_g2b)  # dao nguoc — chinh la meo cho eye-to-hand
        R_b2g.append(T_b2g[:3, :3])
        t_b2g.append(T_b2g[:3, 3])
        R_t2c.append(T_t2c[:3, :3])
        t_t2c.append(T_t2c[:3, 3])

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_t2c, t_t2c, method=method)

    T_cam_to_base = np.eye(4)
    T_cam_to_base[:3, :3] = R_cam2base
    T_cam_to_base[:3, 3] = t_cam2base.ravel()
    return T_cam_to_base


def solve_points(p_base, p_cam):
    """Khớp cứng 2 tập điểm tương ứng (Kabsch): tìm T sao cho p_base ≈ T @ p_cam.
    Trả (T_cam_to_base 4x4, sai số từng điểm [m])."""
    A, B = np.asarray(p_cam, float), np.asarray(p_base, float)
    if len(A) < 4:
        raise ValueError("Cần ít nhất 4 điểm không đồng phẳng, khuyến nghị >= 12.")
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    if S[2] < 1e-4 * S[0]:
        raise ValueError("Các điểm gần như đồng phẳng/thẳng hàng — đổi cả độ sâu lẫn "
                         "ngang/dọc giữa các tư thế rồi thu lại.")
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    res = np.linalg.norm((A @ R.T + T[:3, 3]) - B, axis=1)
    return T, res


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.3, 2.5)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _pose_error(T_true: np.ndarray, T_est: np.ndarray):
    dt = np.linalg.norm(T_true[:3, 3] - T_est[:3, 3])
    dR = T_true[:3, :3].T @ T_est[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
    return dt, ang


def _self_test(arm):
    """Sinh du lieu gia lap: 1 T_cam_to_base 'that' (gia lap) + 1 offset
    marker<->bibut_1 'that' NHƯNG KHÔNG BIẾT (chỉ dùng để sinh dữ liệu, không
    đưa vào solver) — kiểm xem solver có tự tách được offset không cần biết
    trước, và khôi phục đúng T_cam_to_base như thuật toán Tsai-Lenz hứa hẹn."""
    rng = np.random.default_rng(42)

    T_cam_to_base_true = np.eye(4)
    T_cam_to_base_true[:3, :3] = _random_rotation(rng)
    T_cam_to_base_true[:3, 3] = np.array([0.05, -0.10, 0.55]) + rng.uniform(-0.05, 0.05, size=3)

    # Offset marker <-> bibut_1: CO DINH nhung KHONG dua vao solver (dung
    # dung tinh chat "thuat toan tu huy offset qua chuyen dong tuong doi").
    T_marker_offset_true = np.eye(4)
    T_marker_offset_true[:3, :3] = _random_rotation(rng)
    T_marker_offset_true[:3, 3] = rng.uniform(-0.03, 0.03, size=3)

    n_poses = 15
    T_gripper2base_list, T_target2cam_list = [], []
    for _ in range(n_poses):
        servo_degs = rng.uniform(40.0, 140.0, size=len(arm.joint_names)).tolist()
        T_gripper2base = arm.hand_eye_matrix(arm.q_from_servo_degs(servo_degs))
        T_marker2base = T_gripper2base @ T_marker_offset_true
        T_target2cam = invert_T(T_cam_to_base_true) @ T_marker2base
        T_gripper2base_list.append(T_gripper2base)
        T_target2cam_list.append(T_target2cam)

    T_est = solve_hand_eye_eye_to_hand(T_gripper2base_list, T_target2cam_list)
    dt, dang = _pose_error(T_cam_to_base_true, T_est)

    print(f"[self-test] arm={arm.name}, {n_poses} tư thế giả lập, offset marker KHÔNG đưa vào solver.")
    print(f"[self-test] T_cam_to_base thật:\n{T_cam_to_base_true}")
    print(f"[self-test] T_cam_to_base solver tính ra:\n{T_est}")
    print(f"[self-test] Sai số vị trí: {dt*1000:.4f} mm | sai số góc: {dang:.4f}°")

    # Cách 2 (điểm đầu bút): nhiễu đo 3mm mỗi trục, 16 tư thế
    pb, pc = [], []
    T_b2c = invert_T(T_cam_to_base_true)
    for _ in range(16):
        q = arm.q_from_servo_degs(rng.uniform(50.0, 130.0, size=len(arm.joint_names)).tolist())
        tip = np.array(arm.tip_point(q))
        pb.append(tip)
        pc.append(T_b2c[:3, :3] @ tip + T_b2c[:3, 3] + rng.normal(0, 0.003, size=3))
    T_pts, res = solve_points(pb, pc)
    dt2, dang2 = _pose_error(T_cam_to_base_true, T_pts)
    # sai số có ý nghĩa là ở vùng làm việc, không phải ở gốc toạ độ
    work = np.mean(pb, axis=0)
    dwork = np.linalg.norm((T_pts @ T_b2c @ np.append(work, 1))[:3] - work)
    print(f"[self-test] collect-tip (nhiễu 3mm/trục, 16 điểm): lệch góc {dang2:.2f}°, "
          f"lệch tại vùng làm việc {dwork*1000:.2f} mm, dư TB {res.mean()*1000:.1f} mm")
    ok_pts = dwork < 0.004 and dang2 < 3.0

    ok = dt < 1e-3 and dang < 0.1 and ok_pts  # gia lap khong nhieu -> phai gan nhu tuyet doi
    print("[self-test] " + ("ĐẠT — solver khôi phục đúng T_cam_to_base dù không biết offset marker."
                             if ok else "KHÔNG ĐẠT — kiểm tra lại solve_hand_eye_eye_to_hand()."))
    return ok


def load_camera_calib():
    p = CALIB_DIR / "c930e_720p.npz"
    if not p.exists():
        raise SystemExit(f"Chưa có {p} — chạy scripts/calibrate_camera.py trước (Phase 2).")
    d = np.load(p)
    return d["K"].astype(np.float64), d["dist"].astype(np.float64)


def _make_aruco_detector(dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return lambda img: detector.detectMarkers(img)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        return lambda img: cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)


def detect_marker_pose(frame, K, dist, marker_id, marker_size_m, detect_fn):
    """Detect 1 marker ArUco đơn (KHÔNG phải board) và trả về T_target2cam
    (4x4) qua solvePnP trên 4 góc marker với kích thước vật lý đã biết.
    Trả None nếu không thấy đúng marker_id trong frame."""
    corners, ids, _ = detect_fn(frame)
    if ids is None:
        return None
    ids = ids.flatten().tolist()
    if marker_id not in ids:
        return None
    marker_corners = corners[ids.index(marker_id)]

    half = marker_size_m / 2.0
    obj_pts = np.array([
        [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0],
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, marker_corners[0].astype(np.float32), K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def cmd_collect(args, arm):
    """Thu thập cặp (FK pose, marker pose từ camera) qua nhiều tư thế robot
    thật, rồi giải ra T_cam_to_base. CẦN robot + camera + marker vật lý.
    Tự detect marker đơn trực tiếp từ ảnh (không qua vision_aruco_detector)."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Image
    from cv_bridge import CvBridge

    K, dist = load_camera_calib()
    detect_fn = _make_aruco_detector(args.dict)
    marker_size_m = args.marker_size_mm / 1000.0
    bridge = CvBridge()

    rclpy.init()
    node = Node("calibrate_hand_eye_collector")
    state = {"joint": None, "frame": None, "marker_T": None}

    def on_joint(msg: JointState):
        state["joint"] = (list(msg.name), list(msg.position))

    def on_image(msg: Image):
        state["frame"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def on_marker_pose(msg):
        state["marker_T"] = pose_to_T(msg.pose.position, msg.pose.orientation)

    node.create_subscription(JointState, "/pca9685_servo/joint_states", on_joint, 10)
    if args.pose_topic:
        from geometry_msgs.msg import PoseStamped
        from board_detect import pose_to_T
        node.create_subscription(PoseStamped, args.pose_topic, on_marker_pose, 10)
    else:
        node.create_subscription(Image, "/aeroscript/pen_image", on_image, 10)

    print(f"Di chuyển robot qua {args.n_poses} tư thế khác nhau (dùng lệnh "
          f"/pca9685_servo/command ở terminal khác).")
    print(f"Arm: {arm.name}, khớp đọc từ joint_states: {list(arm.joint_names)}")
    print(f"Marker: dict={args.dict}, id={args.marker_id}, cạnh={args.marker_size_mm}mm.")
    print("Ở mỗi tư thế: giữ yên, đợi ảnh ổn định, rồi nhấn Enter ở đây để ghi lại mẫu.")

    T_gripper2base_list, T_target2cam_list = [], []
    try:
        while len(T_gripper2base_list) < args.n_poses:
            input(f"[{len(T_gripper2base_list)}/{args.n_poses}] Enter để ghi mẫu (Ctrl+C để dừng sớm)... ")
            state["marker_T"] = None
            t_end = time.time() + 1.0
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.1)
            if state["joint"] is None or (state["frame"] is None and not args.pose_topic):
                print("  chưa có đủ dữ liệu joint_states + ảnh camera, thử lại.")
                continue
            names, positions = state["joint"]
            try:
                lookup = dict(zip(names, positions))
                servo_degs = [lookup[n] for n in arm.joint_names]
            except KeyError as e:
                print(f"  thiếu khớp {e} trong joint_states, thử lại.")
                continue

            T_target2cam = state.get("marker_T") if args.pose_topic else detect_marker_pose(
                state["frame"], K, dist, args.marker_id, marker_size_m, detect_fn)
            if T_target2cam is None:
                print(f"  KHÔNG thấy marker id={args.marker_id} trong ảnh, thử lại.")
                continue

            T_gripper2base = arm.hand_eye_matrix(arm.q_from_servo_degs(servo_degs))
            T_gripper2base_list.append(T_gripper2base)
            T_target2cam_list.append(T_target2cam)
            print(f"  đã ghi mẫu {len(T_gripper2base_list)}: servo_deg={[f'{d:.1f}' for d in servo_degs]}")
    except KeyboardInterrupt:
        print("\nDừng sớm theo yêu cầu.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if len(T_gripper2base_list) < 3:
        print("Không đủ mẫu để giải (cần >= 3, khuyến nghị 10-15).")
        return 1

    T_cam_to_base = solve_hand_eye_eye_to_hand(T_gripper2base_list, T_target2cam_list)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIB_DIR / "T_cam_to_base.npy"
    np.save(out_path, T_cam_to_base)
    print(f"Đã giải xong với {len(T_gripper2base_list)} mẫu -> {out_path}")
    print(T_cam_to_base)
    return 0


def _auto_tip_targets(arm, centre, span):
    """Lưới điểm đầu bút (base_link, m) quanh `centre`: 3 ngang x 2 cao x 2 sâu
    + tâm, chỉ giữ điểm IK tới được. 'Phía trước' tay là -Y."""
    sx, sy, sz = span
    pts = [tuple(centre)]
    for dy in (-sy / 2, sy / 2):
        for dz in (-sz / 2, sz / 2):
            for dx in (-sx / 2, 0.0, sx / 2):
                pts.append((centre[0] + dx, centre[1] + dy, centre[2] + dz))
    out = []
    for p in pts:
        q = arm.ik_tip(p)
        if q is not None:
            out.append((p, q))
    return out


def cmd_collect_tip(args, arm):
    """Thu cặp điểm (đầu bút theo FK, đầu bút camera đo) rồi khớp ra T_cam_to_base."""
    if arm.name != "newarm":
        raise SystemExit("collect-tip hiện chỉ hỗ trợ --arm newarm.")
    import rclpy
    from geometry_msgs.msg import Point
    from arm_io import ArmIO

    rclpy.init()
    node = rclpy.create_node("calibrate_hand_eye_tip")
    io = ArmIO(node, arm, speed_deg_s=args.speed)
    samples = []
    node.create_subscription(Point, args.tip_topic, lambda m: samples.append((m.x, m.y, m.z)), 10)
    if not io.wait_ready():
        raise SystemExit("Không có /pca9685_servo/joint_states — driver (hoặc cầu nối mô phỏng) chưa chạy.")

    def measure():
        """Trung vị của các mẫu XYZ trong args.dwell giây; None nếu quá ít mẫu."""
        samples.clear()
        io.spin(args.dwell)
        if len(samples) < args.min_samples:
            return None
        return np.median(np.array(samples), axis=0) / 1000.0     # mm -> m

    p_base, p_cam = [], []
    try:
        if args.auto:
            centre = args.centre or [-0.015, arm._fk._J1_WORLD[1] - 0.16, arm._fk._J1_WORLD[2] - 0.163]
            targets = _auto_tip_targets(arm, centre, args.span)
            print(f"Tự đi qua {len(targets)} tư thế quanh {np.round(centre, 3)} m (base_link). Ctrl+C để dừng.")
            for k, (p, q) in enumerate(targets, 1):
                io.move_to(arm.servo_degs_from_q(q))
                m = measure()
                if m is None:
                    print(f"  [{k}/{len(targets)}] camera không thấy bút ở tư thế này — bỏ qua")
                    continue
                p_base.append(arm.tip_point(io.q()))
                p_cam.append(m)
                print(f"  [{k}/{len(targets)}] FK {np.round(np.array(p_base[-1])*1000, 0)} mm | camera {np.round(m*1000, 0)} mm")
            io.move_to(arm.servo_degs_from_q([0.0, 0.0, 0.0, 0.0]))
        else:
            print(f"Đưa tay qua {args.n_poses} tư thế (lệnh /pca9685_servo/command ở terminal khác), "
                  "trải rộng cả ngang, cao và SÂU. Mỗi tư thế giữ yên rồi nhấn Enter.")
            while len(p_base) < args.n_poses:
                input(f"[{len(p_base)}/{args.n_poses}] Enter để ghi mẫu (Ctrl+C dừng sớm)... ")
                m = measure()
                if m is None or io.servo_degs is None:
                    print("  camera không thấy bút (hoặc chưa có joint_states), thử lại.")
                    continue
                p_base.append(arm.tip_point(arm.q_from_servo_degs(io.servo_degs)))
                p_cam.append(m)
                print(f"  đã ghi: FK {np.round(np.array(p_base[-1])*1000, 0)} mm | camera {np.round(m*1000, 0)} mm")
    except KeyboardInterrupt:
        print("\nDừng sớm theo yêu cầu.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    try:
        T, res = solve_points(p_base, p_cam)
    except ValueError as e:
        print(f"Không giải được: {e}")
        return 1
    out_path = Path(args.out) if args.out else CALIB_DIR / "T_cam_to_base.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, T)
    print(f"Đã giải với {len(p_base)} điểm -> {out_path}")
    print(T)
    print(f"Sai số dư: trung bình {res.mean()*1000:.1f} mm, lớn nhất {res.max()*1000:.1f} mm "
          f"(điểm số {int(res.argmax())+1}).")
    print("Dư trung bình dưới ~5mm là dùng được cho bài test vòng kín (goto_board_center.py); "
          "trên 10mm thì kiểm lại home/chiều servo, TOOL_OFFSET và calib camera.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="Kiểm logic bằng số giả lập, không cần phần cứng")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("collect", help="Thu thập dữ liệu thật qua ROS2 rồi giải T_cam_to_base")
    c.add_argument("--n-poses", type=int, default=15)
    c.add_argument("--marker-id", type=int, default=0)
    c.add_argument("--marker-size-mm", type=float, required=True,
                   help="Cạnh marker in ra (mm) — BẮT BUỘC, đo trực tiếp bằng thước sau khi in, "
                        "sai số ở đây tỉ lệ trực tiếp vào mọi khoảng cách 3D tính ra.")
    c.add_argument("--dict", default="DICT_4X4_50",
                   help="Tên dictionary ArUco (vd DICT_4X4_50, DICT_4X4_1000) — phải khớp đúng "
                        "loại đã dùng để tạo marker.")
    c.add_argument("--pose-topic", default=None,
                   help="Đọc pose marker (PoseStamped, hệ camera) từ topic này thay vì tự dò trên "
                        "/aeroscript/pen_image — vd /aeroscript/tool_marker_pose của run_pi4_ros2.py "
                        "(dò trên ảnh thô, không bị bảng chữ overlay che).")
    t = sub.add_parser("collect-tip", help="Không dán marker: dùng XYZ đầu bút camera đo + FK")
    t.add_argument("--auto", action="store_true", help="Tay tự đi qua lưới tư thế (cần bring-up xong)")
    t.add_argument("--n-poses", type=int, default=15, help="[thủ công] số tư thế")
    t.add_argument("--tip-topic", default="/aeroscript/pen_xyz", help="Point, mm, hệ camera")
    t.add_argument("--dwell", type=float, default=1.5, help="giây đứng yên lấy mẫu mỗi tư thế")
    t.add_argument("--min-samples", type=int, default=5)
    t.add_argument("--speed", type=float, default=25.0, help="[auto] tốc độ khớp tối đa (độ/s)")
    t.add_argument("--centre", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="[auto] tâm lưới (m, base_link); mặc định 16cm trước trục J1, ngang tâm bảng")
    t.add_argument("--span", type=float, nargs=3, default=[0.12, 0.08, 0.10], metavar=("X", "Y", "Z"),
                   help="[auto] bề rộng lưới theo ngang / sâu / cao (m)")
    t.add_argument("--out", default=None, help="file .npy ghi kết quả (mặc định calib/T_cam_to_base.npy)")
    add_arm_args(ap)
    args = ap.parse_args()
    arm = arm_from_args(args)

    if args.self_test:
        sys.exit(0 if _self_test(arm) else 1)
    if args.cmd == "collect":
        sys.exit(cmd_collect(args, arm))
    if args.cmd == "collect-tip":
        sys.exit(cmd_collect_tip(args, arm))
    ap.print_help()


if __name__ == "__main__":
    main()
