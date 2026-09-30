# CoVip — Đo vị trí 3D đầu bút trên Raspberry Pi 4

Camera nhìn cây bút → model YOLOv8-pose tìm 4 điểm trên bút (Tip, Tail, L, R) → `solvePnP` tính ra toạ độ X/Y/Z của đầu bút (mm) → publish lên ROS2.

Tài liệu này chỉ gồm các lệnh để **chạy**. Lý do thiết kế và số đo chi tiết xem [PLAN.md](PLAN.md).

---

## 1. Chuẩn bị

### Cần có

| Thứ | Ghi chú |
|---|---|
| Raspberry Pi 4 | Đã cài ROS2 Humble, thư mục code là `~/aeroscript` |
| Camera Logitech (C930e / C920) | Cắm vào cổng **USB 3.0 (viền xanh)** của Pi |
| Laptop (máy dev) | Chứa code gốc tại `~/new_rl_ros2/CoVip` |

### Kết nối

Laptop **kết nối vào wifi do Pi phát**. Địa chỉ của Pi luôn là `192.168.50.1`.

Mở terminal SSH vào Pi:

```bash
ssh piros2@192.168.50.1
```

Mỗi "terminal" trong tài liệu này là **một cửa sổ SSH riêng**, chạy song song, không tắt cái trước khi mở cái sau.

### Cài thư viện trên Pi (chỉ làm 1 lần)

```bash
pip3 install "numpy<2" opencv-python onnxruntime tflite-runtime
sudo apt install ros-humble-web-video-server
```

---

## 2. Đưa code mới nhất lên Pi

Chạy **trên laptop**, mỗi khi code hoặc model trên laptop thay đổi:

```bash
~/new_rl_ros2/CoVip/scripts/deploy_to_pi.sh
```

Chỉ hỏi mật khẩu SSH của Pi **một lần**. Script copy toàn bộ code, model và file calib camera sang `~/aeroscript` trên Pi.

Muốn không phải nhập mật khẩu nữa (làm 1 lần trên laptop): `ssh-copy-id piros2@192.168.50.1`

- Xem trước sẽ copy những gì mà chưa copy thật: `DRY_RUN=1 ~/new_rl_ros2/CoVip/scripts/deploy_to_pi.sh`
- Có file mới cần đưa lên Pi: thêm 1 dòng vào `LOCAL_FILES` và 1 dòng tương ứng vào `REMOTE_FILES` ở đầu script.

---

## 3. Chạy nhận diện bút

> ⚠️ **Không chạy `usb_cam_node_exe` nữa.** Script mới tự mở camera. Nếu `usb_cam` đang chạy, camera sẽ bị chiếm và script báo lỗi không mở được camera.

### Terminal 1 [Pi] — node xử lý ảnh

```bash
cd ~/aeroscript
python3 -u run_pi4_ros2.py --model pen_pose_192_sc.tflite --device /dev/video0 \
    --width 640 --height 360 --fourcc MJPG --conf 0.3 --threads 2 2>&1 | grep -E "⏱️|PEN|Ready|Calib|🔎|❌"
```

Phần `| grep ...` chỉ để lọc log cho gọn. Muốn xem log đầy đủ, bỏ đoạn từ `2>&1` trở đi.

### Terminal 2 [Pi] — web server để xem ảnh

```bash
ros2 run web_video_server web_video_server
```

### Terminal 3 [Pi] — kiểm tra toạ độ ra

```bash
ros2 topic hz /aeroscript/pen_xyz      # tốc độ ra toạ độ thật (Hz)
ros2 topic echo /aeroscript/pen_xyz    # xem X/Y/Z (mm)
```

Nếu báo không tìm thấy lệnh `ros2`: chạy `source /opt/ros/humble/setup.bash` trước.

### Trên laptop — mở trình duyệt

| Mục đích | Link |
|---|---|
| Xem ảnh có vẽ 4 điểm trên bút | http://192.168.50.1:8080/stream?topic=/aeroscript/pen_image&type=mjpeg |
| Trang chọn stream | http://192.168.50.1:8080/stream_viewer?topic=/aeroscript/pen_image |
| Dữ liệu dạng JSON (x, y, z, detected, fps) | http://192.168.50.1:8081/ |

Trên ảnh stream, góc phải hiện **TRACKING** (xanh) khi thấy bút, **NO TARGET** (đỏ) khi không thấy.

**Bảng góc trên-trái — vị trí đầu bút (Tip) so với camera, đơn vị mm.** Gốc toạ độ ở tâm ống kính: **X** dương sang phải, **Y** dương xuống dưới, **Z** dương hướng ra trước camera (khoảng cách theo trục nhìn). `D` là khoảng cách thẳng từ camera tới đầu bút. Chữ xám kèm `[giu cu]` nghĩa là vừa mất bút và đang giữ giá trị cuối.

**Bảng góc dưới-trái — hiệu năng:**

| Dòng | Ý nghĩa |
|---|---|
| `cam->XYZ` | Từ lúc nhận frame từ camera tới lúc publish XYZ. Đây là độ trễ ảnh hưởng tới điều khiển |
| `cam->anh` | Từ lúc nhận frame tới lúc publish ảnh debug (chưa gồm web server và mạng) |
| `= frame cho model A + chay model B` | Tách `cam->XYZ`: A = frame nằm chờ từ lúc nhận tới lúc model bắt đầu chạy, B = thời gian model chạy. Phần còn lại (vài ms) là solvePnP/Kalman/publish |
| `Bat but (2s)` | Tỉ lệ frame thấy bút trong ~2 giây gần nhất |
| `CPU may` / `node` | % cả máy / % của riêng node, tính theo 1 nhân (tối đa 400% trên Pi 4) |
| `RAM node` / `C` | RAM của node / nhiệt độ CPU |

Các mốc E2E tính từ lúc frame tới chương trình, chưa gồm thời gian bên trong camera (phơi sáng, nén, truyền USB, thường thêm ~30-60 ms). Các số này cũng có trong JSON ở port 8081.

### Dừng

Bấm `Ctrl+C` ở từng terminal.

---

## 4. Chọn model và tuỳ chọn

### Các model có sẵn trên Pi

| File | Tốc độ đo thật trên Pi 4 | Ghi chú |
|---|---|---|
| `pen_pose_192_sc.tflite` | ~17 fps (cùng cỡ model 192) | **Khuyên dùng.** Train thêm ảnh bút nghiêng/nằm ngang/chúc xuống/ở xa (30/9): thấy bút 99% khung test, bản 192 cũ 7% |
| `pen_pose_192.tflite` | **~17 fps** | Bản cũ, chỉ nhận tốt bút đứng thẳng và to trong khung |
| `pen_pose_224.tflite` | ~13 fps | Chính xác hơn một chút |
| `pen_pose_192.onnx`, `pen_pose_224.onnx` | chậm hơn ~25-30% | Dùng nếu file `.tflite` báo lỗi khi nạp |
| `best_float32.tflite`, `best.onnx` | ~5 fps | Model cũ (320), chỉ để so sánh |

Đổi model: sửa `--model <tên file>` trong lệnh Terminal 1.

### Các tuỳ chọn hay dùng

| Tuỳ chọn | Ý nghĩa | Nên dùng |
|---|---|---|
| `--model` | File model | `pen_pose_192.tflite` |
| `--device` | Camera | `/dev/video0` |
| `--width --height` | Độ phân giải bắt ảnh | `640 360` (nhanh hơn 1280×720 mà model thấy như nhau) |
| `--fourcc` | Định dạng ảnh camera | `MJPG` |
| `--conf` | Ngưỡng tin cậy, thấp hơn thì dễ nhận hơn nhưng dễ nhận nhầm | `0.3` – `0.55` |
| `--threads` | Số nhân CPU cho model | `2` hoặc `3`. **Không dùng `4`** (chậm hơn vì chiếm hết CPU) |
| `--focus --exposure` | Khoá lấy nét / phơi sáng thủ công | Bỏ trống = tự động |
| `--calib` | File calib camera | `calib/c920_720p.npz` (mặc định). Thiếu file này thì XYZ sai |
| `--trust-motion` | Độ bám của bộ lọc: lớn hơn thì bám nhanh hơn nhưng rung hơn | `1.0`; thấy trễ thì `2`–`4`, thấy rung thì `0.5` |
| `--no-filter` | Tắt bộ lọc, dùng thẳng kết quả từng frame | Chỉ để so sánh độ trễ |
| `--ping-host` | Máy cần đo ping (chỉ có trong JSON port 8081) | Bỏ trống = laptop đang SSH vào Pi |

---

## 5. Đọc log

```
⏱️  [30 frames] pre=7.0ms  invoke=51.6ms  decode=0.5ms  tổng=59.1ms (~16.9 fps thuần model)  conf_max=0.82 (ngưỡng 0.3)
[INFO] ... PEN  X:   61.5  Y:  -33.8  Z:  147.7 mm  FPS:16.5
```

| Mục | Ý nghĩa |
|---|---|
| `pre` / `invoke` / `decode` | Thời gian chuẩn bị ảnh / chạy model / xử lý kết quả, mỗi frame |
| `tổng`, `fps` | Tốc độ xử lý |
| `conf_max` | Độ tin cậy cao nhất model thấy được trong 30 frame. **Gần 0 = model không thấy bút** |
| `PEN X/Y/Z` | Toạ độ đầu bút (mm) so với camera. Chỉ in khi nhận diện thành công |

```
🔎 [75 frame] bắt được 69% | không thấy bút 18 | thiếu keypoint 5 (điểm mờ nhất: L 3, R 2) | PnP lỗi 0
```

Dòng `🔎` in khoảng 5 giây một lần, cho biết vì sao mất nhận diện:
- **không thấy bút**: model không tìm ra bút (bút nghiêng, ở xa, ngoài khung...). Cần cải thiện model/dữ liệu.
- **thiếu keypoint**: thấy bút nhưng 1 trong 4 điểm bị che hoặc mờ, kèm tên điểm hay bị mờ nhất.
- **PnP lỗi**: có đủ 4 điểm nhưng không giải được vị trí 3D.

**Lưu ý:** `FPS` trong log là tốc độ **xử lý** (tính cả frame không thấy bút). Tốc độ **ra toạ độ thật** phải xem bằng `ros2 topic hz /aeroscript/pen_xyz` — con số này thấp hơn nếu bút hay bị mất nhận diện.

---

## 6. Gặp lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| Log báo `❌ KHÔNG tìm thấy file calib` | Chạy lại mục 2 để đưa `calib/c920_720p.npz` lên Pi |
| XYZ đuổi theo bút chậm | Tăng `--trust-motion` (vd `3`), hoặc thử `--no-filter` để so |
| `Không mở được camera` | Kiểm tra camera đã cắm; tắt `usb_cam` nếu đang chạy; thử `--device /dev/video1`. Liệt kê camera: `for d in /sys/class/video4linux/video*; do echo "$d: $(cat $d/name)"; done` |
| `No such file or directory` với file `.py` hoặc model | Chưa đưa code lên Pi → chạy lại mục 2 |
| Lỗi khi nạp file `.tflite` | Dùng bản `.onnx` cùng tên |
| Không có dòng `PEN` nào | Xem `conf_max`: gần 0 → model không thấy bút (thử đổi góc/ánh sáng); nhỉnh hơn ngưỡng một chút → hạ `--conf` |
| Stream trình duyệt không lên | Kiểm tra Terminal 2 đang chạy và laptop đang nối wifi của Pi |
| `ros2: command not found` | `source /opt/ros/humble/setup.bash` |
| Pi chạy chậm bất thường | `htop` xem tiến trình nào đang chiếm CPU |

---

## 7. Đo tài nguyên Pi (CPU, RAM, nhiệt độ)

Mở thêm một terminal SSH trong lúc node đang chạy:

```bash
cd ~/aeroscript
python3 scripts/monitor_resources.py --seconds 60
```

In mỗi giây một dòng, hết giờ in bảng trung bình/cao nhất cho node vision, `web_video_server`, `ros2 topic ...` và cả máy, kèm nhiệt độ CPU và cờ hạ xung. CPU% tính theo 1 nhân: dùng trọn 2 nhân = 200%, Pi 4 tối đa 400%. Thêm `--csv res.csv` để lưu từng mẫu.

## 8. Đo tốc độ riêng model (tuỳ chọn)

Không cần camera, chỉ đo tốc độ chạy model:

```bash
cd ~/aeroscript
python3 scripts/benchmark_tflite.py --model pen_pose_192.tflite --threads 2
python3 scripts/benchmark_onnx.py   --model pen_pose_192.onnx   --threads 2
```

---

## 9. Train lại model (trên laptop, cần GPU NVIDIA)

Chỉ cần khi muốn train lại, ví dụ có dữ liệu mới.

```bash
# Lần đầu: kiểm tra GPU và cài PyTorch bản CUDA vào môi trường train
~/new_rl_ros2/CoVip/imgsz_probe/setup_gpu.sh

# (Chỉ khi dataset gốc thay đổi) tạo lại bộ train/val
~/new_rl_ros2/CoVip/.venv-train/bin/python ~/new_rl_ros2/CoVip/imgsz_probe/make_split.py

# Train (~15-25 phút mỗi lần)
cd ~/new_rl_ros2/CoVip/imgsz_probe
../.venv-train/bin/python train_pose.py --imgsz 192

# Bản có xoay ảnh + phóng/thu khi train: nhận được bút nghiêng, nằm ngang, ở xa
../.venv-train/bin/python train_pose.py --imgsz 192 --degrees 180 --scale 0.8
```

Kết quả: `pen_pose_192.onnx` trong `imgsz_probe/` và `runs/pen_pose_192/weights/best.tflite`. Thêm vào `deploy_to_pi.sh` (mục 2) nếu tên file mới, rồi đưa lên Pi.

### Thêm dữ liệu mới (quay → tách ảnh → gán nhãn → gộp → train)

```bash
cd ~/new_rl_ros2/CoVip
python3 scripts/record_dataset.py record --cam 2 --tag <ten> --focus -1          # quay
python3 scripts/record_dataset.py extract datasets/raw_videos/<ten>_*.avi --every 6 --drop-blur-pct 20
python3 scripts/label_local.py --dirs "<ten>_*" --every 3                        # gán nhãn (click)
python3 scripts/label_local.py --dirs "<ten>_*" --every 3 --merge                # gộp vào dataset_split
cd imgsz_probe && ../.venv-train/bin/python train_pose.py --imgsz 192
```

Không dùng nhãn tự động từ model cũ (`scripts/autolabel.py`) để train khi chưa duyệt từng ảnh. Thử ngày 30/9 trên ảnh có thuyền xanh phía sau: model khoanh nhầm cái thuyền thành bút ở gần hết số ảnh.

---

## 10. File chính

| File | Vai trò |
|---|---|
| `run_pi4_ros2.py` | Node chính chạy trên Pi |
| `scripts/deploy_to_pi.sh` | Đưa code/model lên Pi |
| `scripts/benchmark_tflite.py`, `scripts/benchmark_onnx.py` | Đo tốc độ model |
| `scripts/calibrate_camera.py` | Hiệu chuẩn camera |
| `imgsz_probe/train_pose.py` | Train lại model |
| `dataset_split/` | Dataset đã chia train/val |
| `PLAN.md` | Kế hoạch, số đo, lý do các quyết định |
