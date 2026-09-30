#!/usr/bin/env python3
"""
FK-dự đoán vùng ROI chứa đầu bút trên ảnh (Phase 4 trong PLAN.md) — thay vì
model YOLO phải quét cả khung 1280x720, dùng góc khớp robot đang có (FK)
để đoán trước gần đúng đầu bút nằm ở đâu trên ảnh, chỉ cắt 1 ô nhỏ quanh
đó rồi mới đưa vào model.

Luồng:
    /pca9685_servo/joint_states (độ servo thật, 4 khớp vị trí của tay
    chọn bằng --arm, xem arm_models.py)
      -> quy đổi độ servo -> q (rad, URDF) theo SERVO_SPECS của tay đó
      -> FK -> điểm 3D đầu bút trong hệ base_link
      -> nhân T_cam_to_base (Phase 3, hand-eye calibration) -> hệ camera
      -> chiếu qua K (Phase 2, camera calibration) -> pixel (u, v)
      -> cắt ô ROI_SIZE x ROI_SIZE quanh (u, v), giữ trong biên ảnh

Chạy được ngay bây giờ (không cần Pi/robot/camera thật) để kiểm logic:
    python3 scripts/fk_roi_predictor.py --self-test

Khi có đủ calib/c920_720p.npz (Phase 2) và calib/T_cam_to_base.npy (Phase 3):
    python3 scripts/fk_roi_predictor.py --roi-size 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "calib"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_models import add_arm_args, arm_from_args, joint_states_to_servo_degs  # noqa: E402


def load_calibration():
    """Đọc K/dist (Phase 2) và T_cam_to_base (Phase 3). Trả None nếu chưa có."""
    cam_path = CALIB_DIR / "c920_720p.npz"
    hand_eye_path = CALIB_DIR / "T_cam_to_base.npy"
    K = dist = T_cam_to_base = None
    if cam_path.exists():
        data = np.load(cam_path)
        K, dist = data["K"], data["dist"]
    if hand_eye_path.exists():
        T_cam_to_base = np.load(hand_eye_path)
    return K, dist, T_cam_to_base


def project_point(K: np.ndarray, T_cam_to_base: np.ndarray, point_base_m):
    """Chiếu 1 điểm 3D (m, hệ base_link) qua T_cam_to_base rồi qua K ra pixel
    (u, v). Trả None nếu điểm nằm sau lưng camera (z_cam <= 0)."""
    p_base = np.array([point_base_m[0], point_base_m[1], point_base_m[2], 1.0])
    T_base_to_cam = np.linalg.inv(T_cam_to_base)
    p_cam = T_base_to_cam @ p_base
    x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]
    if z_c <= 1e-6:
        return None
    uv_h = K @ np.array([x_c, y_c, z_c])
    return float(uv_h[0] / uv_h[2]), float(uv_h[1] / uv_h[2])


def predict_roi(arm, servo_degs, roi_size=300,
                 img_w=1280, img_h=720, K=None, T_cam_to_base=None):
    """Trả (u, v, (x0,y0,x1,y1)) — pixel dự đoán + hộp ROI đã cắt trong biên ảnh,
    hoặc None nếu điểm chiếu ra ngoài / sau camera."""
    point_base_m = arm.tip_point(arm.q_from_servo_degs(servo_degs))
    uv = project_point(K, T_cam_to_base, point_base_m)
    if uv is None:
        return None
    u, v = uv
    half = roi_size / 2
    x0 = int(np.clip(u - half, 0, max(img_w - roi_size, 0)))
    y0 = int(np.clip(v - half, 0, max(img_h - roi_size, 0)))
    return u, v, (x0, y0, x0 + roi_size, y0 + roi_size)


def _self_test(arm):
    """Kiểm logic FK -> chiếu -> ROI bằng K và T_cam_to_base giả lập,
    không cần calib thật/phần cứng."""
    K_fake = np.array([
        [900.0, 0.0, 640.0],
        [0.0, 900.0, 360.0],
        [0.0, 0.0, 1.0],
    ])
    # T_cam_to_base giả lập — chỉ để kiểm logic, không phải vị trí camera thật.
    # Tay cũ: đầu bút ở z~0, camera z=0.5 nhìn xuống (-Z; Y_cam đảo theo quy
    # ước). Tay mới (assarm) treo từ z=0.59 xuống, đầu bút ở z~0.1, camera giả
    # lập đặt ở z=-0.4 nhìn LÊN (+Z) để điểm luôn nằm trước camera.
    if arm.name == "assarm":
        T_fake = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.4],
            [0.0, 0.0, 0.0, 1.0],
        ])
    else:
        T_fake = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])

    ok = True
    n_j = len(arm.joint_names)
    for degs_test in ([90.0, 60.0, 120.0, 90.0, 90.0], [90.0] * 5, [60.0, 100.0, 80.0, 110.0, 70.0]):
        degs_test = degs_test[:n_j]
        result = predict_roi(arm, degs_test, K=K_fake, T_cam_to_base=T_fake)
        if result is None:
            print(f"[self-test] servo_deg={degs_test} -> điểm ra sau camera (kiểm tra lại T_cam_to_base giả lập)")
            ok = False
            continue
        u, v, box = result
        in_frame = 0 <= u <= 1280 and 0 <= v <= 720
        print(f"[self-test] servo_deg={degs_test} -> pixel=({u:.1f},{v:.1f}) roi={box}"
              f"  [{'trong khung' if in_frame else 'NGOÀI khung với setup giả lập này'}]")

    # Kiểm sự nhất quán: home (tất cả 90°) phải map ra 1 pixel hợp lệ và
    # trùng khớp khi gọi lại project_point thủ công (không đi qua predict_roi)
    p3d = arm.tip_point(arm.q_from_servo_degs([90.0] * n_j))
    uv_direct = project_point(K_fake, T_fake, p3d)
    r_via_predict = predict_roi(arm, [90.0] * n_j, K=K_fake, T_cam_to_base=T_fake)
    same = (uv_direct is not None and r_via_predict is not None and
            abs(uv_direct[0] - r_via_predict[0]) < 1e-9 and
            abs(uv_direct[1] - r_via_predict[1]) < 1e-9)
    print(f"[self-test] Đối chiếu project_point trực tiếp vs predict_roi: {'KHỚP' if same else 'LỆCH — CÓ LỖI'}")
    ok = ok and same

    print("[self-test] " + ("TẤT CẢ ĐẠT — logic FK->chiếu->ROI chạy đúng." if ok else "CÓ BƯỚC CHƯA ĐẠT, xem log trên."))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="Kiểm logic bằng số giả lập, không cần phần cứng/calib thật")
    ap.add_argument("--roi-size", type=int, default=300)
    add_arm_args(ap)
    args = ap.parse_args()
    arm = arm_from_args(args)

    if args.self_test:
        ok = _self_test(arm)
        sys.exit(0 if ok else 1)

    K, dist, T_cam_to_base = load_calibration()
    if K is None or T_cam_to_base is None:
        missing = []
        if K is None:
            missing.append(str(CALIB_DIR / "c920_720p.npz") + " (Phase 2)")
        if T_cam_to_base is None:
            missing.append(str(CALIB_DIR / "T_cam_to_base.npy") + " (Phase 3)")
        print("Chưa đủ file hiệu chuẩn, cần phần cứng thật để tạo:")
        for m in missing:
            print(f"  - {m}")
        print("Chạy --self-test để kiểm logic trước khi có phần cứng.")
        sys.exit(1)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class FKROIPredictor(Node):
        def __init__(self):
            super().__init__("fk_roi_predictor")
            self.K = K
            self.T_cam_to_base = T_cam_to_base
            self.roi_size = args.roi_size
            self.create_subscription(
                JointState, "/pca9685_servo/joint_states", self._on_joint_states, 10)
            self.get_logger().info(
                f"fk_roi_predictor sẵn sàng, arm={arm.name}, roi_size={self.roi_size}, "
                f"khớp={list(arm.joint_names)}")

        def _on_joint_states(self, msg: JointState):
            try:
                servo_degs = joint_states_to_servo_degs(
                    list(msg.name), list(msg.position), arm.joint_names)
            except ValueError as e:
                self.get_logger().warn(str(e))
                return
            result = predict_roi(
                arm, servo_degs, roi_size=self.roi_size, K=self.K, T_cam_to_base=self.T_cam_to_base)
            if result is None:
                self.get_logger().warn("Điểm dự đoán nằm sau camera — kiểm tra lại T_cam_to_base")
                return
            u, v, box = result
            self.get_logger().info(f"pixel dự đoán=({u:.1f},{v:.1f}) roi={box}")

    rclpy.init()
    node = FKROIPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
