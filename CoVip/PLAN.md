# Plan CHỐT: Tối ưu luồng đo vị trí đầu bút 3D cho tay robot (CoVip)

## 1. Bối cảnh — tối ưu luồng hiện tại, không phải đổi hướng

Luồng hiện tại (YOLOv8-pose quét cả khung hình → PnP ra XYZ) chạy trên Pi 4 rất lag: **7-8 fps, độ trễ 500-1000ms**. Mục tiêu plan này là **tối ưu đúng luồng đó**, không viết lại từ đầu. Có 2 nguyên nhân gây lag, cả 2 sửa được mà không cần train lại model:

1. **Kiến trúc ống dẫn ảnh trên Pi bị nghẽn** — ảnh đi qua `usb_cam` → DDS → `cv_bridge` → node xử lý, mỗi chặng xếp hàng riêng, cộng dồn 500-1000ms dù model chỉ mất ~10ms.
2. **Model phải quét cả khung 1280×720 mỗi lần** để tìm bút — tốn tính toán nhất, và là lý do trước đó phải thu thập dữ liệu cầm tay tự do ở mọi góc/khoảng cách.

Với nguyên nhân (2): robot đã tự biết gần đúng bút đang ở đâu, nên không cần quét cả khung nữa —

- `fk_ik_utils.py` đã có hàm `fk(q)` tính chính xác vị trí `bibut_1` từ góc khớp, thẳng từ URDF.
- Topic `/pca9685_servo/joint_states` đã publish góc khớp thật.
- `vs_lib/vision/vision_aruco_detector.py` đã có sẵn node ArUco dùng để đo đạc.
- `config/T_cam_to_base_THEORETICAL.npy` chỉ là số đoán tay, **chưa từng đo thật**.

→ Dùng FK để **thu hẹp vùng xử lý** xuống 1 ô nhỏ (thay vì cả khung) vừa giảm tải tính toán, vừa xoá luôn nhu cầu gán tay hàng nghìn ảnh cầm tay tự do.

**Việc gán nhãn 260 ảnh cầm tay đang làm dở: dừng hẳn.** Video cầm tay cũ (5 phiên, 1945 ảnh) không dùng làm dữ liệu train chính nữa — lý do chi tiết ở Phase 6.

## 2. Tiến độ (cập nhật 2026-09-23)

| Phase | Việc                                                    | Trạng thái                                          | Cần gì để làm tiếp                              |
| ----- | -------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| 1     | Sửa kiến trúc ống dẫn ảnh Pi (`run_pi4_ros2.py`) | ✅ Đã deploy + chạy + đo thật trên Pi (xem mục 5b)  | Đã tối ưu hết mức phần mềm (~4.8→ước tính 6.5-7fps). Trần cứng 8fps ở`imgsz=320` → muốn 15-20fps phải train lại `imgsz=224` (mục 5c) |
| —    | `fk_4dof()`/`fk_4dof_matrix()` cho robot 4-DOF       | ✅ Xong, verify khớp tuyệt đối với`fk()` 6-DOF | Chỉ còn dùng cho tay cũ (`--arm old4dof`)      |
| —    | **Chuyển sang tay mới `assarm`** (mục 2b)          | ✅ FK + script vision + config driver xong, self-test đạt | Xác nhận trên robot thật: home/chiều servo, bản 180°/270°, kênh PCA9685, đo `TOOL_OFFSET` |
| 2     | `scripts/calibrate_camera.py`                          | ✅ Chạy thật OK với C930e (test), lỗi 0.24px<0.5px | **Chạy LẠI với C920 thật khi có** (K/dist riêng theo từng camera vật lý, số hiện tại chỉ để test pipeline) |
| 3     | `scripts/calibrate_hand_eye.py`                        | ✅ Code xong (viết lại dùng marker đơn thật), `--self-test` khớp tuyệt đối | Marker đã in xong (DICT_4X4_50 id0), cần robot để `collect` |
| 4     | `scripts/fk_roi_predictor.py`                          | ✅ Code xong,`--self-test` chạy đạt              | `calib/c920_720p.npz` + `calib/T_cam_to_base.npy` |
| 5     | Node ghép hoàn chỉnh publish`/pen_xyz`              | ⬜ Chưa viết                                        | Phase 4 chạy được với calib thật trước        |
| 6     | Dataset mới (bút gắn tay robot, camera cố định)    | ⬜ Tuỳ chọn, không chặn                           | Chỉ cần nếu Phase 5 đo chưa đủ chính xác     |
| —    | Train lại model ở`imgsz=224` để đạt 15-20fps    | ⬜ Đang làm — dataset đã có (2445 ảnh)          | Cài driver NVIDIA để dùng RTX 3060 (mục 5c); đo tốc độ từng `imgsz` trước khi train |
| sau   | Bù độ nén lò xo (bài toán cân bằng drone)       | ⬜ Cố tình chưa làm                               | Sau khi Phase 1-5 ổn định                          |

3 script mới tự kiểm ngay bằng số giả lập, không cần phần cứng:

```bash
cd /home/ducanh/new_rl_ros2/CoVip
python3 scripts/fk_roi_predictor.py --self-test
python3 scripts/calibrate_hand_eye.py --self-test
```

## 2b. Chuyển sang cánh tay mới `assarm` (2026-09-30)

Thiết kế mới: `ref/assarm_ikfk-20260930T054840Z-1-001/assarm_ikfk/` (URDF xuất từ Fusion 360, package **ROS1**). Đầu bút + giá bút giữ nguyên như cũ.

| Khớp | URDF | Servo | Trục | Vai trò |
| --- | --- | --- | --- | --- |
| J1 base | `Revolute_Joint_1` | TD-8120MG | (0,0,-1) | yaw |
| J2 shoulder | `Revolute_Joint_2` | RDS3120 | (-1,0,0) | pitch |
| J3 elbow | `Revolute_Joint_3` | MG995 | (-1,0,0) | pitch |
| J4 wrist_roll | `Revolute_Joint_4` | MG995 | (0,0,-1) | xoay quanh trục cẳng tay |
| J5 | `Revolute_Joint_Active` | MG995 | (0,1,0) | servo gripper/giá bút, ngoài chuỗi vị trí |

**Khác tay cũ:** không còn `wrist_pitch` → vị trí đầu bút chỉ do J1-J3 quyết định, hướng bút = hướng cẳng tay (q2+q3), **không chỉnh độc lập được**. Ở q=0 tay treo thẳng xuống từ z=0.59m (Servo1) tới z=0.197m (Plate_1); đoạn J2→J3 dài 0.142m, J3→J4 dài 0.166m.

**Lỗi trong URDF export đã sửa (trong FK, không sửa file URDF gốc):** `Revolute_Joint_1` có origin lệch 0.54m khỏi tay → quay J1 là tay văng ra xa. Đối chiếu tâm bánh răng trong STL (`Servo1_Pinion_Gear_1`: (0, -0.0108, 0.5458)) và dời trục về đúng chỗ. J2/J3/J4 khớp STL. Cụm Servo5/gripper cũng bị đặt sai (z≈0) nhưng không ảnh hưởng vision. **Nếu port URDF này sang ROS2/Gazebo cho phần RL, phải sửa lỗi J1 này trong URDF.**

**Đã làm:**
- `ros2_ws/.../rl/fk_assarm.py` — FK thuần Python (`fk_flange_matrix`, `fk_tip`), `SERVO_SPECS` (home/range/đảo chiều từng servo), `TOOL_OFFSET`. Verify: J1=0 khớp URDF gốc tới 1e-16 ở 200 bộ góc ngẫu nhiên; J1 quay đúng quanh trục bánh răng. `fk_ik_utils.py` KHÔNG đổi (phần RL còn dùng).
- `CoVip/scripts/arm_models.py` — chọn tay bằng `--arm assarm|old4dof` (mặc định `assarm`). `fk_roi_predictor.py` và `calibrate_hand_eye.py` đã chuyển sang dùng nó; `--self-test` đạt với cả 2 tay. Hand-eye của `assarm` dùng frame **Plate_1** (không phụ thuộc `TOOL_OFFSET`).
- `wicom_roboarm/config/servos_assarm.yaml` + launch arg: `ros2 launch wicom_roboarm wicom_roboarm.launch.py servo_config:=servos_assarm.yaml`. Tên khớp: `base, shoulder, elbow, wrist_roll, j5`.

**Việc cần xác nhận trên robot thật (trước Phase 3):**
1. **Bản 180° hay 270°** của RDS3120/TD-8120MG. Driver map lệnh 0-180 tuyến tính vào 500-2500µs; bản 270° sẽ quay 1.5 lần → sửa `range_deg` trong `SERVO_SPECS` (hoặc thu hẹp dải xung trong yaml), không thì FK sai 50%.
2. **Home + chiều quay:** gửi 90° cả 4 khớp → tay phải treo thẳng xuống (= URDF q=0). Tăng lệnh từng khớp, so với chiều URDF (J1 +q quay theo chiều kim đồng hồ nhìn từ trên xuống quanh -Z, J2/J3 +q quay quanh -X). Ngược thì đặt `inverted: True`.
3. **Kênh PCA9685** thật cho 5 servo (yaml đang để tạm 0/1/4/7/15).
4. **Đo `TOOL_OFFSET`** (vector Plate_1 → đầu bút, hệ Plate_1) bằng thước kẹp. Hiện ước lượng `(0, 0, -0.10)`. Chỉ ảnh hưởng ROI (Phase 4), truyền qua `--tool-offset X Y Z` hoặc sửa hằng số.
5. **Sai số MG995:** servo analog, không phản hồi vị trí thật (`joint_states` là góc LỆNH), có rơ/xệ khi chịu tải. Mỗi 1° sai ở J2 ≈ 6mm ở đầu bút → chọn `--roi-size` rộng rãi, và FK chỉ dùng để khoanh ROI, toạ độ cuối vẫn lấy từ vision.

**IK + khả năng vẽ lên bảng ArUco thẳng đứng (2026-09-30):**
- `fk_assarm.ik_tip()` / `ik_tip_nearest()` — IK giải tích (J1-J3 vị trí, J4 cố định), khớp FK tới 1e-12 trên 5000 tư thế, ~10µs/lần. `ik_tip_nearest` giữ nhánh khi bám liên tục.
- `ros2_ws/.../rl/assarm_board_reach.py` — quét vị trí bảng (hình vuông 10cm + nhấc bút 2cm, không cho khuỷu/cổ tay chạm bảng).
- Kết quả (TOOL_OFFSET ước lượng): bảng phải cách trục J1 **~32-37cm**, tâm hình **~5-20cm dưới trục J1**; góc bút-pháp tuyến ~15-25° (tốt hơn tilt 35° của tay cũ). Vùng vẽ được **rất hẹp** vì khuỷu chỉ ±90°: rung sâu ±2cm vẫn bám được, ±3cm mất IK 4%, ±5cm mất 17%.
- **Nếu lắp lệch home servo khuỷu để có q3 ∈ [-30°,150°]**: vùng vẽ rộng gấp ~4 lần (bảng d≈27cm, z≈37cm), rung sâu ±7cm vẫn 0% mất IK. Nên cân nhắc khi lắp ráp (servo 180° chỉ đổi được VỊ TRÍ dải, không nới được dải).
- RL cũ (`drawing_config.X_PLANE=-0.50`, fallback board z=0.60) KHÔNG áp dụng được cho tay mới — tầm với chỉ ~0.41m từ vai.

**Bút + J5 + giới hạn cơ khí (2026-09-30):**
- Giá bút cũ lắp **thay gripper, trên trục servo J5**, trục J5 → đầu bút **56mm**. Vị trí trục J5 lấy từ STL (`J5_IN_PLATE`, ước lượng — đo lại khi lắp). Tên khớp J5 = `pen`.
- **J4 = ±90° thì trục J5 song song J2/J3 → J5 là wrist pitch**: tay mới tương đương tay 4-DOF cũ (base, shoulder, elbow, wrist_pitch; J4 giữ cố định).
- Quét va chạm STL: vai về phía trước chỉ tới **+82°** (URDF ghi +90), khuỷu gập được **-110°..+142°** → khuyến nghị lắp sừng servo khuỷu cho cửa sổ **[-40°, 140°]** (lệnh 40° = cẳng tay thẳng).
- `assarm_board_reach.py` (giới hạn trên): J5 khoá q5=0 → 57 vị trí vẽ được, góc bút tốt nhất 19°; **J5 làm wrist pitch (`--j5 wrist`) → 79 vị trí, góc bút ~10-17° ở hầu hết vùng**, chịu lệch cao/thấp của drone tốt hơn nhiều. → Nên vẽ ở chế độ wrist.
- Vision: `arm_models.AssArm` đọc 5 khớp (có `pen`); hand-eye dùng khung giá bút sau J5 (`fk_pen_matrix`), ROI dùng đầu bút theo góc J5 thật.
- `assarm_dh_table.py` — bảng Standard DH (J1-J3, J5 khoá) + bộ số để đối chiếu công cụ ngoài.

**Chưa làm (ngoài phạm vi vision):** phần RL/digital twin (`control_backends.py` `GAZEBO_TO_PI_JOINT_MAP` 6 khớp, `drawing_environment.py`, Gazebo URDF) vẫn là tay cũ; URDF mới là ROS1 cần port sang ROS2 + sửa lỗi J1.

## 3. Ghi chú kỹ thuật cố định: FK cho robot 4-DOF (tay CŨ — `--arm old4dof`)

Robot thật dùng biến thể 4-DOF: servo 4 (wrist_roll) và servo 6 (pen) bị khoá cứng, không di chuyển — 6-DOF gốc còn lại 4 khớp điều khiển được (base/shoulder/elbow/wrist_pitch).

`fk_4dof()`/`fk_4dof_matrix()` trong `ros2_ws/src/visual_servoing/scripts/rl/fk_ik_utils.py` tái dùng NGUYÊN chuỗi transform thật của bản 6-DOF (KHÔNG phải mô hình lượng giác phẳng rút gọn L1/L2/L3 của `wicom_roboarm_4dof_standalone.py` — đó là hệ số đo tay cũ, không dùng ở đây), chỉ khoá cứng 2 góc theo đúng ánh xạ vật lý servo-độ↔rad trong `control_backends.py` (`GAZEBO_TO_PI_JOINT_MAP`, home=90° cho cả 6 khớp). Mặc định khoá ở **90°/90°** (= vị trí mặc định mô phỏng). Đã verify khớp tuyệt đối với `fk()` gốc ở nhiều bộ góc.

**Nếu robot thật khoá KHÁC vị trí mặc định mô phỏng:** mọi lệnh gọi `fk_4dof(...)`/các script Phase 3-5 bên dưới cần truyền thêm góc thật, ví dụ `fk_4dof(q4, wrist_roll_deg=<góc thật>, pen_deg=<góc thật>)` hoặc cờ `--wrist-roll-deg`/`--pen-deg`. Xác nhận việc này ở Bước 0.

## 4. Chuẩn bị trước khi bắt đầu — Bước 0

Quy ước ký hiệu máy chạy lệnh dùng xuyên suốt phần dưới:

- **[PI]** = SSH vào Raspberry Pi (`ssh piros2@192.168.50.1`) — cần cho I2C (servo), camera vật lý, hoặc topic ROS2 chỉ tồn tại trên Pi.
- **[LAPTOP]** = chạy trên máy dev đang ngồi, không cần SSH.
- Mỗi "Terminal N [MÁY]" là 1 cửa sổ SSH/local riêng, chạy song song — không tắt cửa sổ trước khi mở cửa sổ sau, trừ khi ghi rõ.

### 4a. Đồng bộ code lên Pi (bắt buộc, làm trước MỌI bước [PI] khác)

Toàn bộ code trong plan này (Phase 1 sửa `run_pi4_ros2.py`, `fk_4dof()`, 4 script mới) hiện chỉ nằm trên máy dev, **chưa hề có trên Pi**. Vài điểm cần biết:

- `CoVip` là git repo RIÊNG (remote `origin` → `DoanCuong2401/DA1_EmbedLab.git`), đang có rất nhiều thay đổi CHƯA commit (cả đợt xoá file rác cũ lẫn toàn bộ script mới — đều untracked). **Pi KHÔNG có repo này** (xem xác nhận bên dưới) nên hướng đồng bộ qua git không áp dụng được — dùng scp.
- **File trùng tên gây nhầm lẫn:** `CoVip/pen_models/run_pi4_ros2.py` (máy dev) là bản CŨ (kiến trúc `usb_cam`→`/image_raw` trước Phase 1). Luôn lấy từ `CoVip/run_pi4_ros2.py` (thư mục gốc) để đưa lên Pi, KHÔNG phải bản trong `pen_models/`.

**✅ ĐÃ XÁC MINH (2026-09-23, qua `ls ~` thật trên Pi):** thư mục vision trên Pi tên **`~/aeroscript`**, cấu trúc **PHẲNG** — `best.onnx`, `run_pi4_ros2.py` nằm thẳng trong đó, KHÔNG có thư mục con `pen_models/` như máy dev (đã có sẵn 1 `run_pi4_ros2.py` bản CŨ ở đó, `deploy_to_pi.sh` sẽ ghi đè bằng bản Phase 1 mới). Không thấy git repo — đây không phải clone của `DA1_EmbedLab`, chỉ là thư mục file thuần. `~/ros2_ws` là workspace ROS2 riêng, nằm cùng cấp `~/aeroscript` dưới home — chưa xác nhận có sẵn `visual_servoing` package trong đó không (chỉ cần cho Phase 3/4, không cần cho Phase 1).

`deploy_to_pi.sh` đã được cập nhật đúng theo cấu trúc thật này — map riêng từng file (`CoVip/X` máy dev → `aeroscript/X` trên Pi, `ros2_ws/Y` → `ros2_ws/Y` giữ nguyên tên):

```bash
/home/ducanh/new_rl_ros2/CoVip/scripts/deploy_to_pi.sh
```

Không cần truyền `PI_HOST`/`PI_HOME` gì thêm — mặc định `piros2@192.168.50.1` + home Pi (`~`) đã đúng. Xem trước sẽ copy gì mà chưa copy thật: `DRY_RUN=1 .../deploy_to_pi.sh`. **Dùng lại về sau:** có file mới cần port thì thêm 1 cặp dòng vào `LOCAL_FILES`/`REMOTE_FILES` đầu script rồi chạy lại.

**Lưu ý khi tự chạy:** tôi (Claude) không có mật khẩu/khoá SSH vào Pi của bạn nên không tự chạy `scp`/`ssh` thật được — bạn cần tự chạy lệnh trên trong terminal của mình (sẽ được hỏi mật khẩu SSH bình thường).

### 4b. Xác nhận vật lý

- Nhìn/đo trực tiếp góc servo 4 (wrist_roll) và servo 6 (pen) đang khoá cứng trên robot thật, so với mục 3 ở trên.
- **[PI]**: cắm C920 vào cổng USB 3.0 (viền xanh) của Pi 4, không dùng USB 2.0.

## 5. Phase 1 — Sửa kiến trúc ống dẫn ảnh trên Pi — ✅ ĐÃ XONG, đã chạy & đo thật trên Pi (2026-09-23)

**Luồng cũ** (theo `ROS2 - How to run(2) (1).md`): `usb_cam_node_exe` (`pixel_format:="yuyv"`, ảnh thô KHÔNG nén, 640×480/15fps) → publish `/image_raw` qua DDS → `run_pi4_ros2.py` subscribe bằng `cv_bridge`. Mỗi chặng xếp hàng riêng, cộng dồn 500-1000ms.

**Đã sửa:** bắt frame bằng OpenCV trực tiếp trong cùng tiến trình với node xử lý (bỏ hẳn `usb_cam`→DDS→`cv_bridge`), luôn lấy frame mới nhất, drop frame cũ nếu xử lý chưa xong. Đổi định dạng bắt ảnh từ YUYV thô sang **MJPEG** (YUYV ở 720p ~28MB/s dễ nghẽn USB, MJPEG nén sẵn nhẹ hơn nhiều lần). Ảnh debug tách thành luồng phụ riêng, không chặn luồng chính publish XYZ.

**Độ phân giải chọn: 1280×720 (720p) MJPEG**, không lên 1080p — đủ chi tiết cho model detect trong ROI nhỏ (Phase 4), trong khi 1080p tốn gần gấp đôi CPU Pi 4 không cần thiết. **Camera Logitech C920 + Pi 4 Model B**: hoạt động tốt, C920 hỗ trợ MJPEG sẵn tới 1080p/30fps, Pi 4 Model B có 2 cổng USB 3.0 dư băng thông.

**Terminal 1 [PI]** — chạy node xử lý ảnh chính:

```bash
cd ~/aeroscript
python3 run_pi4_ros2.py --model best.onnx --device /dev/video0 \
    --width 1280 --height 720 --fourcc MJPG --conf 0.55
```

**Terminal 2 [PI]** (SSH thêm cửa sổ mới) — đo kết quả:

```bash
ros2 topic hz /aeroscript/pen_xyz      # đo fps thật
ros2 topic echo /aeroscript/pen_xyz    # xem toạ độ XYZ (mm)
```

**Terminal 3 [PI]** (tuỳ chọn, xem ảnh debug qua trình duyệt):

```bash
ros2 run web_video_server web_video_server
```

Sau đó **[LAPTOP]**: mở `http://192.168.50.1:8080/stream_viewer?topic=/aeroscript/pen_image`.

Ghi lại fps/latency ở Terminal 2 — mục tiêu cải thiện rõ so với 500-1000ms/7-8fps cũ.

## 5b. Kết quả đo thật trên Pi 4 + giới hạn tốc độ (2026-09-23)

### Đã đo được gì

| Cấu hình | FPS xử lý | Ghi chú |
| --- | --- | --- |
| ONNX `best.onnx`, threads=1 | ~2.8 | đo trước Phase 1 |
| ONNX, threads=2 | 4.0-4.4 | khoảng trống mất detect tối đa 5.6s |
| ONNX, threads=3 | ~4.3 | |
| ONNX, threads=4 | 3.5-3.8 | **tệ hơn** threads=2 — chiếm hết 4 core, bỏ đói FrameGrabber/ROS executor; khoảng trống tối đa 28s |
| TFLite `best_float32.tflite`, threads=2, 720p | 4.6-5.6 | |
| TFLite, threads=2, 640×480 | 5.8-6.0 | giảm độ phân giải **gần như không giúp** |

**Benchmark thuần suy luận** (`scripts/benchmark_tflite.py`, cô lập, threads=4): `best_float32.tflite` mean **115.6ms** (~8.6fps); `best_float16.tflite` mean 190ms — **fp16 CHẬM HƠN fp32** vì Cortex-A72 không có phần cứng ARMv8.2-FP16, phải quy đổi ngược lúc chạy. Không dùng bản fp16.

### Ngân sách thời gian mỗi frame (đo trong pipeline thật, TFLite threads=2, 720p)

```
pre=28ms   invoke=176-189ms   decode=1ms   →  tổng 206ms  (~4.8 fps)
```

### Giới hạn cứng — KHÔNG đạt được 15-20fps bằng tối ưu phần mềm

Bản thân `invoke` đã tốn **116ms** trong điều kiện lý tưởng (benchmark cô lập, không gì chạy cùng) → trần tuyệt đối ≈ **8fps** ở `imgsz=320` trên CPU Pi 4. Mục tiêu 15fps đòi hỏi ≤66ms/frame, tức phải nhanh hơn model hiện tại gần 2 lần **ngay cả khi mọi phần khác bằng 0**. Chỉ có 3 đường thật sự tới đích, không có đường thứ 4:

1. **Giảm `imgsz` 320 → 224** (FLOPs × 0.49, ước tính ~12-15fps) — xem mục 5c.
2. Thêm phần cứng tăng tốc (Coral USB TPU cắm USB3 của Pi 4).
3. Đổi nền tảng (Pi 5 + AI HAT, hoặc Jetson).

### Đã thử và LOẠI TRỪ (đừng thử lại)

- **Giảm độ phân giải camera** (720p→480p): chỉ tăng ~10% (5.0-5.6 → 5.8-6.0). Lý do: model luôn resize ảnh về `imgsz` cố định nên chi phí suy luận **không đổi** theo độ phân giải đầu vào.
- **Chuyển việc vẽ overlay sang thread phụ**: FPS thô gần như không đổi. (Vẫn giữ vì đúng về mặt kiến trúc, và `/pen_xyz` hz có cải thiện.)
- **Viết lại bằng C++**: không đáng — phần tính toán nặng của ONNX/TFLite vốn đã là C++ biên dịch sẵn, Python chỉ gọi vào.
- **`best_int8.onnx`**: lượng tử hoá INT8 làm **mất hẳn khả năng detect** bút (hỏng keypoint nhỏ như mép nắp bút). Không dùng.
- **Hardware JPEG decode của Pi 4**: Pi 4 (BCM2711) có giải mã cứng H.264/HEVC nhưng **KHÔNG có mã hoá cứng** (Pi 3 có, Pi 4 bỏ). Có khối JPEG decode trong ISP lộ qua V4L2 M2M (`bcm2835-codec`) nhưng `cv2.VideoCapture` không tự dùng — phải dựng pipeline GStreamer và OpenCV phải build kèm GStreamer (bản `pip install opencv-python` **không có**). Chưa làm; chỉ đáng làm nếu đo thấy `retrieve()` tốn nhiều.

### Đã sửa trong `run_pi4_ros2.py` (2026-09-23)

1. **Warmup model lúc khởi động** — lần gọi đầu tiên luôn chậm bất thường (JIT + graph optimization chạy lúc đó, không phải lúc load). Trước khi sửa, frame camera thật đầu tiên gánh luôn chi phí này → một lần trễ 26-50s lúc khởi động.
2. **Backend TFLite + XNNPACK** (`TFLitePoseInference`), chọn tự động theo đuôi file `--model` (`.tflite` → TFLite, còn lại → ONNX). Nhanh hơn ONNX ~25-30% (đo đúng cùng điều kiện — số "~2x" ghi ở bản trước là SAI, đem TFLite đo cô lập so với ONNX đo trong pipeline thật, hai điều kiện khác nhau).
   **Bug đã gặp và đã sửa:** export TFLite này xuất toạ độ box/keypoint **đã chuẩn hoá về [0,1]**, khác ONNX xuất thẳng pixel-space [0,320] → nếu không nhân lại `imgsz`, `solvePnP` ra vị trí sai hàng chục mét. Sửa bằng `_COORD_COLS` trong `_normalize_output()`. Tìm ra bằng `scripts/inspect_tflite.py` (output `min=-0.1583 max=1.0113` là dấu hiệu rõ).
3. **`--threads` mặc định 1 → 3** (lý do cũ "nhường CPU cho usb_cam" không còn đúng từ Phase 1). Thực tế nên dùng **2 hoặc 3, tuyệt đối không dùng 4**.
4. **Tiền xử lý dùng buffer cấp phát sẵn** — bản cũ cấp phát ~2.7MB mỗi frame (`canvas` + `.astype()` + `/255.0` tạo 2 mảng float 1.2MB) và đổi BGR→RGB bằng slice stride âm `[:, :, ::-1]` (buộc copy ngược, chậm). Nay ghi thẳng vào buffer có sẵn + `cv2.cvtColor`. Verify khớp bản cũ trong sai số 1 ULP float32 (nhân nghịch đảo thay vì chia) ở cả NHWC/NCHW, 3 độ phân giải; nhanh 3.6x trên máy dev.
5. **Tách `grab()` khỏi `retrieve()` trong `FrameGrabber`** — trước đây `read()` giải mã JPEG hết tốc độ camera (~30fps) trong khi vòng xử lý chỉ dùng ~5 frame/giây → **~25 lần giải mã bị vứt đi mỗi giây**, đốt CPU và băng thông bộ nhớ mà XNNPACK cần. Nay `grab()` (rẻ, không giải mã) chạy liên tục để giữ frame luôn mới, chỉ `retrieve()` khi vòng chính gọi `request_next()`.
6. **Đo thời gian từng giai đoạn** — in `⏱️ pre/invoke/decode` mỗi 30 frame và `📷 retrieve()` mỗi 30 lần giải mã, để chẩn đoán bằng số đo thật thay vì đoán.

### Lưu ý đọc số: "FPS" trong log ≠ tốc độ ra kết quả

Con số `FPS:` in trong log là **tốc độ xử lý frame**, đếm mọi lần chạy pipeline bất kể có detect được bút hay không. Còn `ros2 topic hz /aeroscript/pen_xyz` chỉ đếm lúc **detect thành công** — thực đo chỉ **1-2.8 Hz** với khoảng trống tới 11-19s. Chênh lệch này là **vấn đề độ tin cậy phát hiện, không phải vấn đề tốc độ** — đúng thứ Phase 4 (ROI theo FK) sinh ra để sửa. Tăng FPS thô không sửa được nó.

## 5c. Giảm `imgsz` 320 → 224: train lại (2026-09-23)

**Không tìm thấy `best.pt` gốc** — `archive/export_tflite.py` trỏ sang máy khác (`/home/luongduy/AeroScript_Vision/...`), quét toàn bộ `/home/ducanh` không có file YOLO `.pt` nào.

**Nhưng dataset gốc còn đủ:** `CoVip/COVIP_training.v4i.yolov8/` — **2445 ảnh + 2445 nhãn**, `kpt_shape: [4, 3]`, `nc: 1` (`pen_tip`) — khớp chính xác model đang chạy. Nên train lại từ dataset, không cần đi tìm file cũ.

Lưu ý `data.yaml`: `path:` đang trỏ sang máy khác (`/home/doancuong/...`) cần sửa thành đường dẫn thật; `val: train/images` (chưa tách tập validation riêng).

### Kết quả kiểm tra dataset (2026-09-23)

**Chất lượng gán nhãn: rất tốt** — ghép cặp ảnh/nhãn đủ 2445/2445; trung điểm L/R nằm ở 0.692 trên trục Tip→Tail trong khi `PEN_3D` kỳ vọng 0.688 (lệch 0.6%, phân tán hẹp p5=0.655/p95=0.739); thứ tự L/R nhất quán 100% (1860/1860); 0 keypoint ngoài khung; chỉ 1.6% nhãn nghi ngờ. 588 nhãn rỗng = ~196 ảnh gốc **thật sự không có bút** (đã xem tận mắt xác nhận), hợp lệ và có ích để giảm báo nhầm.

**Nhưng có 3 vấn đề:**

1. **SAI BỐI CẢNH TRIỂN KHAI (nặng nhất).** Ảnh thật trong dataset: bút **cầm trên tay**, quay lia trong phòng lộn xộn, **tay người luôn trong khung**. Triển khai thật: bút gắn cứng trên cánh tay robot, camera cố định, không có tay người. Model chưa từng thấy cấu hình thật. → **Đây là gốc rễ của việc mất detect 10-20s trên Pi, KHÔNG phải do tốc độ.** Train lại bằng chính dữ liệu này ở `imgsz` nào cũng không sửa được. Đúng như mục 11 đã chốt từ trước: dữ liệu cầm tay không dùng làm dữ liệu train chính.
2. **Chỉ 815 ảnh gốc**, không phải 2445 — Roboflow nhân 3 bản/ảnh nhưng phép nhân bản chỉ đổi độ sáng ±10% + nhiễu muối tiêu 0.1% (không xoay/đổi tỉ lệ/dịch chuyển), gần như không thêm đa dạng. Và **không có tập validation** → không đo được học vẹt.
3. **Ảnh train bị bóp méo hình**: Roboflow *"Resize to 640x640 (Stretch)"* phá tỉ lệ khung, trong khi `run_pi4_ros2.py` lúc chạy lại letterbox giữ tỉ lệ → bút có hình dạng khác giữa train và chạy thật.

**Nghi vấn `PEN_3D` sai kích thước:** tỉ lệ `|L-R|/|Tip-Tail|` đo từ nhãn = **0.302**, còn `PEN_3D` trong `run_pi4_ros2.py` giả định **0.359** (23mm/64mm) — lệch 16%. Nếu `PEN_3D` sai thì `solvePnP` cho **Z sai hệ thống** theo đúng tỉ lệ đó. **Cần lấy thước kẹp đo lại bút thật**: Tip→Tail có đúng 64mm, bề ngang chỗ L/R có đúng 23mm không.

### Kết luận: đủ hay chưa?

| Mục đích | Đủ chưa |
| --- | --- |
| Train lại`imgsz=224` để **tăng tốc**, chất lượng ngang model hiện tại | ✅ Đủ — bút chiếm trung vị 49% chiều cao khung; ở 224 vẫn còn 109px (p5=77px), thừa sức detect |
| Sửa việc **mất detect 10-20s** | ❌ Không đủ — sai bối cảnh; phải quay dataset mới theo mục 11, hoặc dùng Phase 4 (ROI theo FK) để né |

Hai việc này **độc lập nhau**: `imgsz=224` giải quyết tốc độ, Phase 4 / dataset mới giải quyết độ tin cậy.

**Máy dev có RTX 3060 Mobile** nhưng đang chạy driver `nouveau` (không hỗ trợ CUDA) → `torch.cuda.is_available()` = False dù torch đã là bản `2.6.0+cu124`. Secure Boot đã tắt, driver khuyến nghị `nvidia-driver-595-open`:

```bash
sudo ubuntu-drivers autoinstall && sudo reboot
# sau khi khởi động lại:
nvidia-smi && python3 -c "import torch; print(torch.cuda.is_available())"
```

Có GPU: train ~15-25 phút. Không GPU (16 core CPU): ~5-12 tiếng.

**✅ Driver đã cài xong (2026-09-23):** `nvidia-driver-595-open` + module kernel khớp `6.8.0-138-generic`, `nvidia-prime` chọn chế độ **on-demand** (màn hình chạy AMD iGPU, RTX 3060 dành cho tính toán — an toàn cho laptop đồ hoạ lai). **Phải khởi động lại máy** thì `nouveau` mới nhả ra cho module `nvidia` nạp vào.

Sau khi khởi động lại, chạy **một lệnh** để hoàn tất (kiểm tra driver + thay `torch` bản CPU trong venv bằng bản CUDA + kiểm tra lại):

```bash
./CoVip/imgsz_probe/setup_gpu.sh
```

**Thứ tự làm (đo trước, train sau):**

1. **Đo trước khi train** — tốc độ suy luận chỉ phụ thuộc kiến trúc + `imgsz`, **không phụ thuộc trọng số**. Lấy `yolov8n-pose` gốc của COCO export ở 320/256/224/192, đem lên Pi benchmark → biết con số thật của từng `imgsz` trong ~15 phút, thay vì train 8 tiếng rồi mới biết có đáng không. (Model COCO có 17 keypoint thay vì 4 → chênh lệch tốc độ <2%, vẫn đại diện tốt.)
2. Chọn `imgsz` theo số đo thật.
3. Train `yolov8n-pose` trên 2445 ảnh ở `imgsz` đã chọn, export `.tflite` float32.
4. Deploy + so độ chính xác với model hiện tại (bản ONNX 320 hiện tại đã chính xác — đây là mốc so sánh).

Dùng môi trường Python **riêng biệt** cho việc này, không cài `ultralytics` vào `.venv` của repo (tránh nó đổi phiên bản numpy/torch làm hỏng setup RL).

### Đã dựng sẵn (2026-09-23)

- **`CoVip/.venv-train/`** — môi trường riêng, `ultralytics 8.4.160`. ⚠️ Nó tự kéo về `torch 2.9.1+cpu`, **phải thay bằng bản CUDA** sau khi cài driver thì mới dùng được RTX 3060.
- **`CoVip/dataset_split/`** — dataset đã tách val đúng cách: **2073 ảnh train / 119 ảnh val**, dùng symlink (không nhân đôi ảnh). Gom nhóm theo (tiền tố tên, nội dung nhãn) để 3 bản augment của cùng ảnh gốc luôn nằm cùng một phía → **không rò rỉ dữ liệu**; val chỉ lấy 1 bản/nhóm.
  **Đã sửa `flip_idx` [0,1,2,3] → [0,1,3,2]**: Ultralytics mặc định lật ngang 50% số frame, mà nhãn quy ước L = mép TRÁI trong ảnh (đúng 100% ở cả 1860 nhãn), nên với `[0,1,2,3]` thì ~50% số frame bị **dạy sai L/R**. Nhiều khả năng đây chính là lỗi L/R hoán đổi đã ghi ở mục 11.
- **`CoVip/imgsz_probe/make_split.py`** — script tạo bản tách val ở trên.
- **`CoVip/imgsz_probe/train_pose.py`** — train + tự export, tự nhận GPU. Chạy: `../.venv-train/bin/python train_pose.py --imgsz 224`
- **`CoVip/imgsz_probe/probe_{320,256,224,192}.onnx`** + **`CoVip/scripts/benchmark_onnx.py`** — đo tốc độ từng `imgsz` trên Pi trước khi train (đã thêm vào `deploy_to_pi.sh`).

**FLOPs theo `imgsz`** (yolov8n-pose, lấy từ log export): 320 ≈ 2.34 GFLOPs (suy ra) / 256 = 1.5 / **224 = 1.1 (47% so với 320)** / 192 = 0.8 (34%).

### ✅ ĐÃ ĐO THẬT trên Pi (2026-09-23, `benchmark_onnx.py`, threads=2)

| imgsz | ONNX đo được | Tỉ lệ so với 320 | FLOPs dự đoán |
| --- | --- | --- | --- |
| 320 | 194.7ms | 1.00 | 1.00 |
| 256 | 121.9ms | 0.63 | 0.64 ✓ |
| **224** | **101.5ms** | **0.52** | 0.47 |
| 192 | 74.4ms | 0.38 | 0.34 |

Rất ổn định (max-min chỉ 1-2ms), co giãn gần đúng theo FLOPs. Áp tỉ lệ này lên `invoke` TFLite (152ms ở 320) + `pre` 27ms + `decode` 1ms:

| imgsz | invoke dự phóng | Tổng/frame | FPS | FPS nếu camera 640×360 (`pre`~10ms) |
| --- | --- | --- | --- | --- |
| 320 | 152ms | 180ms | **5.6** (đo thật 5.4-5.6 ✓) | — |
| 256 | 95ms | 123ms | **8.1** | — |
| 224 | 79ms | 107ms | **9.3** | **~11** |
| 192 | 58ms | 86ms | **11.6** | **~14.5** |

Dòng 320 khớp số đo thật → phần dự phóng đáng tin.

**Kết luận:** 224 → ~10-11fps (gấp đôi hiện tại, chưa tới 15); 192 → ~14-15fps (chạm mục tiêu dưới); **20fps ngoài tầm với model này trên CPU Pi 4**.

**Đánh đổi:** độ chính xác định vị keypoint giảm theo đúng tỉ lệ thu nhỏ (224 kém 1.43x, 192 kém 1.67x so với 320) → sai số XYZ từ `solvePnP` tăng tương ứng. Khả năng *phát hiện* không lo: bút chiếm trung vị 49% chiều cao khung, ở 192 vẫn còn 94px.

→ **Train cả 224 và 192** (mỗi lần ~20 phút trên RTX 3060) rồi đo cả tốc độ lẫn sai số thật trên Pi để chọn, thay vì đoán.

**⚠️ Đường export TFLite đang hỏng:** `ultralytics 8.4.160` bỏ `format='tflite'`, chuyển sang `format='litert'` dùng `ai-edge-torch` vốn đòi **torch ≥ 2.11** (venv có 2.9.1) → lỗi `cannot import name 'ScalingType' from 'torch.nn.functional'`. Tạm thời dùng **ONNX** cho việc đo tốc độ. Model cuối cùng vẫn nên có bản TFLite (nhanh hơn ONNX ~25-30% trên Pi, không phải ~2x — xem mục 5b) — khi đó chọn 1 trong: nâng torch ≥2.11, hoặc hạ ultralytics về 8.3.x (đường export cũ qua `onnx2tf`, đã có sẵn `tensorflow 2.20` trong venv). **`train_pose.py` đã sửa để export ONNX trước (luôn chạy được), TFLite thử sau và bắt lỗi gọn nếu hỏng** — không chặn việc có model dùng được ngay sau khi train.

## 6. Phase 2 — Calibrate camera — ✅ Đã chạy thật với C930e (test), lỗi 0.24px

**Kết quả chạy thật (2026-09-23), camera C930e:** 28/28 ảnh dùng được, reprojection error **0.2422px** (đạt tốt so với mục tiêu <0.5px). File lưu ở `calib/c920_720p.npz`.

**⚠️ Đây là camera TEST (C930e), không phải C920 thật sẽ deploy.** Ma trận K/dist là đặc trưng riêng của từng camera vật lý cụ thể (kể cả 2 máy cùng model cũng khác nhau) — **bắt buộc chạy lại đúng quy trình này với chính con C920 thật** khi có, không dùng lại số vừa ra cho deployment thật. Kết quả hiện tại chỉ có giá trị để: (a) xác nhận toàn bộ pipeline code (`capture`→`compute`→dùng ở Phase 3/4) chạy đúng, (b) tập luyện quy trình chụp trước khi làm với hàng thật.

Calibrate trên máy host (laptop) là được, không cần Pi — ma trận K/dist chỉ phụ thuộc camera+lens+độ phân giải. Điều kiện bắt buộc khi làm với C920 thật: **đúng camera vật lý C920** sẽ gắn lên Pi, **đúng 720p MJPEG** sẽ deploy, **khoá focus cố định trước khi chụp, không đổi lại sau đó** (lỡ chạm phải calibrate lại từ đầu).

**[LAPTOP hoặc PI]** (ví dụ dưới đây trên LAPTOP, đỡ SSH):

```bash
cd /home/ducanh/new_rl_ros2/CoVip
# In bàn cờ 10x7 ô (9x6 góc trong), dán phẳng lên bìa cứng

# 2a. Chụp ảnh (SPACE lưu khi thấy khung xanh bọc quanh bàn cờ, q để thoát)
# KHÔNG dùng --device /dev/video0 nếu máy có nhiều camera (laptop thường có
# sẵn webcam tích hợp ở video0) — mặc định script tự tìm đúng camera rời
# theo tên thiết bị (--camera-name, mặc định "C930e" khi đang test; đổi
# thành "C920" khi chuyển sang camera deploy thật):
python3 scripts/calibrate_camera.py capture --camera-name C930e \
    --width 1280 --height 720 --focus 20 --cols 9 --rows 6 \
    --out calib/chessboard_raw
# -> chụp 15-20 ảnh, nhiều góc nghiêng/khoảng cách khác nhau
# Không chắc tên thiết bị? Liệt kê: for d in /sys/class/video4linux/video*; do echo "$d: $(cat $d/name)"; done

# 2b. Tính K/dist
python3 scripts/calibrate_camera.py compute --images calib/chessboard_raw \
    --cols 9 --rows 6 --square-mm 25 --out calib/c920_720p.npz
```

Kiểm log: `Reprojection error trung bình: X.XXXXpx` — cần **< 0.5px**, thấp hơn thì chụp thêm ảnh đa dạng góc rồi chạy lại 2b.

**Nếu chụp trên LAPTOP:** copy sang Pi:

```bash
scp calib/c920_720p.npz piros2@192.168.50.1:~/aeroscript/calib/
```

Từ giờ không được xoay/chạm focus camera nữa.

## 6b. Làm được ngay với CHỈ camera + bút, chưa cần robot/Pi

Trước khi cần đến robot (Phase 3+), có 2 việc kiểm tra trước giúp tránh mất công dựng cả robot rồi mới phát hiện lỗi:

**Kiểm marker thật có detect được không** (`scripts/test_marker_detection.py`, mới viết) — xác nhận đúng dict/id/size trước khi cần robot:
```bash
cd /home/ducanh/new_rl_ros2/CoVip
python3 scripts/test_marker_detection.py --camera-name C930e \
    --marker-id 0 --marker-size-mm 10 --dict DICT_4X4_50
```
Cửa sổ hiện lên, đưa marker vào khung hình — thấy % detect tăng lên + vẽ 3 trục toạ độ lên marker là đúng. Marker 10mm khá nhỏ, tầm detect ổn định chắc chỉ trong khoảng ~15-30cm — nếu quá xa mà không detect được, cần biết trước để tính lại khoảng cách làm việc khi calibrate thật ở Phase 3.

**Kiểm giả thuyết cốt lõi Phase 4** (`scripts/test_roi_detection.py`) — xem detect trong ô ROI nhỏ có tốt hơn quét cả khung không, trả lời trước câu hỏi "có cần Phase 6 (retrain) hay không":
```bash
python3 scripts/test_roi_detection.py --camera-name C930e --roi-size 300
```
Mũi tên di chuyển ô ROI, `q` thoát in báo cáo % detect full-frame vs % detect trong ROI.

## 7. Phase 3 — Hand-eye calibration — ✅ Script xong (solver + self-test), chưa chạy thật

**Phát hiện quan trọng khi viết solver:** bố trí thật là "eye-to-hand" (camera cố định trên khung, marker gắn trên phần di chuyển của tay) — NGƯỢC với "eye-in-hand" mặc định của `cv2.calibrateHandEye()`, nên phải đưa nghịch đảo của gripper2base vào hàm mới ra đúng `T_cam_to_base` (kỹ thuật chuẩn cho eye-to-hand). Đã kiểm bằng `--self-test`: solver khôi phục đúng tuyệt đối `T_cam_to_base` dù KHÔNG biết offset thật giữa marker và `bibut_1` — đúng tính chất thuật toán Tsai-Lenz (dùng chuyển động tương đối giữa các tư thế để tự triệt tiêu offset không biết trước, miễn offset đó cố định suốt quá trình đo). Do đó điều kiện bắt buộc: **Tip và đĩa gắn marker phải cố định cứng với nhau** — chỉ Tail dịch khi ép lò xo.

**Marker thật đã làm (2026-09-23):** 1 marker ArUco ĐƠN, dictionary **DICT_4X4_50**, ID **0**, in dán trực tiếp lên đĩa cứng ngay sát điểm gắn bút (xem ảnh) — khoảng cách tâm marker → đầu bút đo được **56mm** (chỉ để tham khảo/đối chiếu sau này, KHÔNG cần nhập vào solver — do tính chất Tsai-Lenz ở trên). Bố trí này **khác với dự tính ban đầu** (dán marker lên mặt phẳng cánh tay, dùng board 4-marker của `vision_aruco_detector.py` có sẵn) — do đó **không dùng node `vision_aruco_detector` nữa**: nó viết cho board 4 marker `DICT_4X4_1000` (cần thấy ≥2 marker cùng lúc), khác hẳn dictionary và bố cục 1-marker-đơn đang dùng. `calibrate_hand_eye.py` đã được viết lại để **tự detect marker đơn trực tiếp** (đọc ảnh từ `/aeroscript/pen_image` + K/dist từ `calib/c920_720p.npz`), không phụ thuộc node/topic nào của package `visual_servoing` nữa — nhờ vậy Phase 3 giờ chỉ còn **4 terminal thay vì 6** (bỏ hẳn terminal publish_camera_info + terminal vision_aruco_detector).

- Cần đo lại chính xác **cạnh marker in ra** (mm, không phải khoảng cách tới đầu bút) bằng thước — bắt buộc truyền đúng qua `--marker-size-mm`, sai số ở đây tỉ lệ trực tiếp vào mọi khoảng cách 3D tính ra sau này.
- Cần Phase 2 xong trước (cần `calib/c920_720p.npz` để giải PnP marker chính xác).
- Cần robot di chuyển được qua lệnh (`/pca9685_servo/command`) để lấy nhiều tư thế.

Toàn bộ 4 terminal dưới đây mở trên **[PI]**, chạy song song, không tắt cửa sổ trước khi mở cửa sổ sau:

**Terminal 1 [PI]** — node xử lý ảnh (dùng lại y hệt Phase 1, để có sẵn `/aeroscript/pen_image`):

```bash
cd ~/aeroscript
python3 run_pi4_ros2.py --model best.onnx --device /dev/video0 \
    --width 1280 --height 720 --fourcc MJPG
```

**Terminal 2 [PI]** — driver servo (node đứng sau `/pca9685_servo/command` và `/pca9685_servo/joint_states`, theo đúng README gốc mục "Set Home"):

```bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py servo_config:=servos_assarm.yaml   # tay mới
```

**Terminal 3 [PI]** — di chuyển tay robot qua từng tư thế (tay mới `assarm`: 4 khớp `base/shoulder/elbow/wrist_roll`, giữ `j5` cố định; tay cũ: `base/shoulder/elbow/wrist_pitch`, KHÔNG gửi wrist_roll/pen). Với tay mới, nhớ đổi dòng `wrist_pitch` bên dưới thành `wrist_roll`:

```bash
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[70.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['shoulder'], position:[110.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['elbow'], position:[95.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_pitch'], position:[100.0]}"
```

Đổi 4 con số thành 1 tư thế mới mỗi lần lặp lại — dàn trải khắp vùng làm việc an toàn, không dồn về 1 góc.

**Terminal 4 [PI]** — công cụ thu thập mẫu + giải hand-eye (thay `<cạnh_marker_mm>` bằng số đo thật):

```bash
cd ~/aeroscript
python3 scripts/calibrate_hand_eye.py collect --n-poses 15 \
    --marker-id 0 --marker-size-mm <cạnh_marker_mm> --dict DICT_4X4_50
```

Quy trình lặp 15 lần: (a) Terminal 3 gửi 4 lệnh set góc cho 1 tư thế mới → (b) đợi robot dừng hẳn + marker hiện rõ trong khung hình → (c) Terminal 4 nhấn Enter ghi mẫu (script tự báo nếu chưa thấy marker, không ghi mẫu lỗi) → lặp lại (a).

Xong đủ 15 mẫu, Terminal 4 tự giải và lưu `calib/T_cam_to_base.npy`. Nếu nghi ngờ sai số: chiếu `fk_4dof(q)` qua `T_cam_to_base` ra ảnh, so với vị trí marker thật, lệch phải **< 1-2cm**.

Xong bước này: Ctrl+C tắt Terminal 1-2, **gỡ marker khỏi robot** — không dùng khi vận hành thật.

## 8. Phase 4 — ROI dự đoán bằng FK, vẫn detect bằng model YOLO — ✅ Script xong, chưa chạy thật

### Phase 4 giải quyết đúng vấn đề gì?

Nhắc lại vấn đề gốc từ mục 1: model `best.onnx` hiện có được train chủ yếu trên ảnh **cận cảnh** (bút chiếm phần lớn khung hình). Khi đưa cả khung 1280×720 (bút chỉ chiếm vài % diện tích ảnh, lẫn trong nền phòng/tay người/ánh sáng lộn xộn) vào thẳng model, độ chính xác giảm mạnh — đây chính là "domain gap" đã đo được trước đó (~24.7% detect thành công trên ảnh cầm tay full-frame, so với ~98%+ trên ảnh cận cảnh lúc train). Retrain lại model với hàng nghìn ảnh full-frame là 1 cách sửa (Phase 6), nhưng tốn công và vẫn không giải quyết vấn đề tốc độ.

**Phase 4 giải quyết vấn đề bằng cách khác: không để model phải "tìm" bút trong cả khung hình nữa — cho nó biết trước gần đúng bút ở đâu.** Cụ thể pipeline `fk_roi_predictor.py` làm:

1. Đọc góc 4 khớp hiện tại của robot từ `/pca9685_servo/joint_states` (robot luôn biết chính nó đang ở tư thế nào).
2. Tính `fk_4dof(q)` → ra toạ độ 3D thật của đầu bút (`bibut_1`) trong hệ toạ độ robot (mét) — đây là hình học cơ khí thuần tuý, không liên quan gì đến ảnh/camera.
3. Nhân với `T_cam_to_base` (kết quả Phase 3) để đổi toạ độ 3D đó sang hệ toạ độ của camera.
4. Chiếu điểm 3D (hệ camera) qua ma trận nội tại `K` (kết quả Phase 2) → ra đúng 1 toạ độ pixel (u, v) trên ảnh 1280×720 — đây chính là **dự đoán trước bút sẽ xuất hiện ở đâu trên ảnh**, tính toán thuần bằng hình học, hoàn toàn không cần chạy model hay nhìn ảnh.
5. Cắt 1 ô nhỏ ~300×300px quanh (u, v) đó.

**Kết quả mang lại — giải quyết đồng thời CẢ 2 vấn đề gốc nêu ở mục 1:**
- **Tốc độ/độ trễ:** model chỉ phải xử lý 1 ô 300×300px thay vì cả khung 1280×720 (diện tích giảm ~10 lần) → suy luận nhanh hơn nhiều lần, đây là phần tối ưu độ trễ lớn nhất trong toàn bộ plan.
- **Độ chính xác:** trong ô 300×300px đã biết trước bút nằm gần đó, bút sẽ chiếm tỉ lệ lớn của ảnh crop — đúng loại điều kiện ảnh cận cảnh mà `best.onnx` đã học tốt, không cần train lại model ngay.

**Giới hạn của Phase 4 (tự thân nó CHƯA làm):** `fk_roi_predictor.py` hiện tại chỉ tính toán và IN RA pixel dự đoán + toạ độ ô ROI — nó **KHÔNG tự cắt ảnh, KHÔNG chạy model detect, KHÔNG tính ra toạ độ 3D cuối cùng của bút**. Việc chạy thật Phase 4 chỉ nhằm mục đích **kiểm tra bằng mắt xem dự đoán vị trí có đúng không** (ô ROI có luôn bọc quanh đầu bút thật hay bị lệch). Phần cắt ảnh + chạy `best.onnx` trong ô đó + xuất toạ độ `/pen_xyz` cuối cùng là việc của **Phase 5** (chưa viết) — Phase 4 là bước chuẩn bị/xác nhận "định vị trước" hoạt động đúng, trước khi ghép nốt phần detect+publish vào.

**Bỏ hướng lọc màu** — nếu sau này đổi cánh tay/vỏ bút khác màu, pipeline lọc màu cứng sẽ hỏng ngay. Dùng lại model detect (`best.onnx`, giống `pen_webcam_onnx.py`) làm bộ nhận diện chính trong ROI — học đặc điểm hình dạng, không phụ thuộc màu.

Đã kiểm logic FK→chiếu→ROI bằng `--self-test` (số giả lập, không cần calib/phần cứng thật) và chạy đạt.

**Terminal 1 [PI]** — bật driver servo (nếu chưa chạy):

```bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

**Terminal 2 [PI]**:

```bash
cd ~/aeroscript
python3 scripts/fk_roi_predictor.py --roi-size 300
```

**Terminal 3 [PI]**: di chuyển robot qua vài tư thế (giống Terminal 3 ở Phase 3).

Ở Terminal 2, xem log `pixel dự đoán=(...)` — đối chiếu bằng mắt với ảnh thật (`/aeroscript/pen_image` qua web_video_server), kiểm ô ROI có luôn bọc đúng quanh đầu bút không.

## 9. Phase 5 — Ghép thành node hoàn chỉnh — ⬜ Chưa viết, làm SAU khi Phase 4 xác nhận ROI đúng

Việc còn lại: gộp `fk_roi_predictor.py` (crop ROI) + model detect trong ROI (tái dùng logic từ `pen_webcam_onnx.py`/`run_pi4_ros2.py`) + publish `/pen_xyz` (mm, hệ base_link) — chạy trên nền luồng Pi đã sửa ở Phase 1. Z lấy từ FK theo hình học cứng, **chưa cộng phần lệch do lò xo** (xem mục 10) — chấp nhận sai số vài mm-cm ở giai đoạn này, đủ để có pipeline chạy được và đo tốc độ/độ trễ thật. Báo lại kết quả Phase 4 để làm tiếp phần này.

## 10. Phase sau (không làm bây giờ) — Đo tinh độ nén lò xo

Đây là drone/UAV cầm bút — khi ép bút vào mặt phẳng sẽ phát sinh phản lực khiến drone phải tự cân bằng lại, không đơn thuần "đo khoảng lệch rồi cộng vào Z" như tay robot cố định trên bàn. Cần xét cùng bài toán điều khiển cân bằng drone — để lại làm **sau khi Phase 1-5 chạy ổn định**, không chặn tiến độ hiện tại.

## 11. Phase 6 — Cải thiện model trong ROI (tuỳ chọn, không chặn)

**⚠️ Lỗi đã ghi nhận (2026-09-23), CHƯA SỬA — để xử lý khi làm Phase 6:** test bằng `test_roi_detection.py` cho thấy model detect đúng cả 4 điểm (specs đạt), nhưng **2 điểm Left/Right bị lẫn/sai chiều** so với hướng thật của bút trong ảnh — theo hình, Left phải nằm bên trái marker, Right phải nằm bên phải marker, nhưng model ra ngược. Cần kiểm lại khi đánh giá model ở Phase 6 (có thể do quy ước gán nhãn train trước đây không nhất quán, hoặc do hướng cầm bút khi test ngược chiều lúc train).

Trước mắt dùng thẳng `best.onnx` hiện có để detect trong ROI — ảnh crop nhỏ khiến bút chiếm tỉ lệ lớn, gần giống điều kiện model đã học, có thể chưa cần train lại ngay.

**Toàn bộ video cầm tay cũ (5 phiên, 1945 ảnh) + việc gán nhãn 260 ảnh dở dang: dừng hẳn, không dùng làm dữ liệu train chính nữa.** Dữ liệu đúng phải quay theo đúng bối cảnh triển khai — bút gắn trên cánh tay robot, camera ở tư thế cố định như treo trên drone khi bay (không lắc), chỉ thay đổi bằng cách cho robot di chuyển qua nhiều tư thế khớp.

**Chỉ làm nếu Phase 4/5 đo ra chưa đủ chính xác trong ROI**, hoặc muốn chuẩn bị trước (không bắt buộc, không chặn Phase 1-5). **[LAPTOP hoặc PI]**, camera phải gắn cố định đúng tư thế thật (không cầm tay), đã khoá focus/exposure như Phase 2:

```bash
cd /home/ducanh/new_rl_ros2/CoVip
python3 scripts/record_dataset.py record --tag robot_mounted \
    --width 1280 --height 720 --focus 20
# Trong lúc quay [PI, terminal khác]: cho robot chạy qua nhiều tư thế bằng lệnh
# ros2 topic pub .../pca9685_servo/command (giống Phase 3 Terminal 3), có lúc ép/thả lò xo.
# Dừng quay bằng Ctrl+C ở terminal đang record.

python3 scripts/record_dataset.py extract datasets/raw_videos/robot_mounted*.avi \
    --every 6 --min-diff 2.0 --drop-blur-pct 20
```

Dữ liệu này chỉ dùng khi thật sự cần cải thiện model, chưa cần gán nhãn/train ngay.

## 12. File chính liên quan

- Mới: `CoVip/scripts/{calibrate_camera,calibrate_hand_eye,fk_roi_predictor,test_roi_detection,deploy_to_pi.sh}`, `CoVip/calib/*.npz|*.npy` (tạo ra ở Phase 2/3)
- Mới (chẩn đoán hiệu năng Pi, mục 5b): `CoVip/scripts/benchmark_tflite.py` (đo tốc độ suy luận thuần, cô lập), `CoVip/scripts/inspect_tflite.py` (in layout output thật của file `.tflite` — dùng để tìm bug toạ độ chuẩn hoá [0,1])
- Dataset train lại (mục 5c): `CoVip/COVIP_training.v4i.yolov8/` — 2445 ảnh + nhãn 4 keypoint
- Đã sửa: `CoVip/run_pi4_ros2.py` (Phase 1), `ros2_ws/src/visual_servoing/scripts/rl/fk_ik_utils.py` (thêm `fk_4dof`, `fk_matrix`, `fk_4dof_matrix`)
- Đọc/tái dùng, không sửa: `config/T_cam_to_base_THEORETICAL.npy` (sẽ thay bằng bản đo thật), `CoVip/scripts/pen_webcam_onnx.py`, `CoVip/pen_models/best.onnx`
- **Không dùng:** `CoVip/pen_models/run_pi4_ros2.py` (bản cũ, xem mục 4a), `vs_lib/vision/vision_aruco_detector.py` (viết cho board 4-marker khác dictionary, xem mục 7), `CoVip/scripts/publish_camera_info.py` (chỉ cần khi dùng `vision_aruco_detector`; `calibrate_hand_eye.py` giờ đọc thẳng `calib/c920_720p.npz`, không cần topic `/camera_info` nữa — giữ lại file phòng khi cần publish camera_info cho việc khác)

## 13. Verification — tiêu chí đạt của từng Phase

- **Phase 1:** độ trễ đầu-cuối trên Pi giảm rõ rệt so với 500-1000ms/7-8fps ban đầu (chưa cần đạt mục tiêu cuối).
- **Phase 2:** reprojection error của `cv2.calibrateCamera` < 0.5px.
- **Phase 3:** chiếu `fk_4dof(q)` qua `T_cam_to_base` ra ảnh, lệch so với marker thật < 1-2cm ở vài tư thế kiểm tra.
- **Phase 4+5:** đặt bút ở khoảng cách/độ nén biết trước (đo tay bằng thước), so với kết quả pipeline, sai số mục tiêu vài mm.
- **Sau Phase 5:** đo lại độ trễ đầu-cuối trên Pi lần cuối, mục tiêu ≤150ms, fps ≥15.
