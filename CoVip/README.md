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

**Chữ trên ảnh stream** — tiếng Anh, chữ trắng viền đen, mỗi vật một dòng, đơn vị mm, số nguyên:

```
PEN (mm) cam X+10 Y-5 Z410 | board x+2 y-1 | 12 mm from board | inside draw area
BOARD (mm) cam X+8 Y-4 Z400 | markers seen 0 1 2 3 | tilt 35 deg | fit error 0.5 px
15.0 FPS | delay 92 ms | pen found 98% of frames | CPU 55% | 62 C          (góc dưới)
```

| Phần | Ý nghĩa |
|---|---|
| `PEN … cam X Y Z` | Đầu bút so với camera: gốc ở tâm ống kính, **X** sang phải, **Y** xuống dưới, **Z** ra trước ống kính |
| `board x y` | Đầu bút chiếu xuống mặt bảng: gốc ở dấu +, **x** sang phải, **y** lên trên (chỉ có khi chạy `--board`) |
| `12 mm from board` | Đầu bút cách mặt bảng 12 mm theo phương vuông góc (0 = chạm bảng) |
| `inside / OUTSIDE draw area` | Đầu bút nằm trong hay ngoài hình vuông vùng vẽ 100×100 mm |
| `PEN not found` (đỏ) | Không thấy bút |
| `BOARD … cam X Y Z` | Dấu + giữa bảng so với camera |
| `markers seen 0 1 2 3` | ID các marker lần dò gần nhất còn thấy. `markers hidden (pose 7 s old)` = đang bị che, dùng vị trí bảng nhớ từ 7 giây trước |
| `tilt 35 deg` | Bảng nghiêng 35° so với hướng nhìn thẳng của camera (0 = nhìn chính diện) |
| `fit error 0.5 px` | Vị trí bảng tính ra lệch các góc marker trên ảnh trung bình 0,5 pixel (càng nhỏ càng tốt) |
| `delay 92 ms` | Từ lúc frame tới chương trình đến lúc phát XYZ (chưa gồm ~30-60 ms bên trong camera) |
| `pen found 98% of frames` | Tỉ lệ frame thấy bút trong ~2 giây gần nhất |
| `CPU` / `C` | CPU cả máy / nhiệt độ CPU |

Hình vẽ trên ảnh đều nét mảnh để không che bút: 4 chấm điểm khớp của bút kèm tên, 3 trục của bút dài 15 mm. Số liệu chi tiết hơn (độ trễ từng chặng, RAM, …) nằm trong JSON port 8081 và trong log lượt chạy (mục 7).

Ảnh stream luôn rộng 640 px: bắt ảnh 1280×720 thì node thu nhỏ trước khi gửi, nên stream không nặng thêm. Đổi bằng `--stream-width` (0 = gửi đúng cỡ bắt ảnh).

### Đo đầu bút so với mặt phẳng vẽ (bảng 4 marker ArUco)

Thêm `--board` vào lệnh Terminal 1. Camera dò 4 marker của bảng vẽ để xác định mặt phẳng; đưa bút vào là có toạ độ đầu bút **so với bảng**.

```bash
python3 -u run_pi4_ros2.py --model pen_pose_192_sc.tflite --device /dev/video0 \
    --width 640 --height 360 --fourcc MJPG --conf 0.3 --threads 2 --board 2>&1 | grep -E "PEN|🎯|Calib|🔎|❌|📝"
```

Bảng in: `ros2_ws/src/visual_servoing/aruco_markers/workspace_board_newarm_A4.pdf` (A4, in 100%, marker 30 mm). In cỡ khác thì thêm `--board-marker-mm` và `--board-offset-mm` theo số đo thật.

**Hệ toạ độ bảng (mm):** gốc ở dấu **+** giữa bảng. **x** sang phải, **y** lên trên (khi nhìn vào bảng). **Cách mặt bảng** là khoảng cách vuông góc từ đầu bút tới mặt bảng: 0 là chạm bảng, số dương là ở phía trước bảng.

| Xem ở đâu | Nội dung |
|---|---|
| Log | `PEN … \| so với BẢNG x:+12.3 y:-4.5 cách mặt bảng: 31.0 mm (trong vùng vẽ)` |
| Ảnh stream | Phần `board x… y… \| … mm from board` trên dòng `PEN`, và dòng `BOARD`. Khung vùng vẽ 100×100 mm (lục = bút ở trong, cam = ở ngoài); 2 trục bảng ở dấu + (đỏ = x, lục = y); chấm tím = hình chiếu đầu bút xuống bảng. **Bám bảng:** ô quanh mỗi marker kèm số ID — lục = lần dò gần nhất còn thấy, đỏ = đang bị che (vị trí suy từ lần thấy trước); chấm vàng = góc marker dò được thật |
| Topic | `ros2 topic echo /aeroscript/pen_board_xyz` (Point, mm). Pose bảng: `/aeroscript/board_pose` |
| JSON port 8081 | `pen_board_x/y/z`, `board_cam_x/y/z`, `board_tilt_deg`, `board_seen_ids`, `board_reproj_px`, `board_age_s`, `aruco_ms` |

Cách dùng cho đúng:
- Để camera thấy **đủ 4 marker** trước khi đưa bút vào (log `🎯 Dò ArUco: thấy bảng N/20`, N gần 20). Sau đó tay hoặc bút che bớt marker cũng không sao: node nhớ vị trí bảng.
- **Camera và bảng nên đứng yên lúc đo.** Dời một trong hai thì node bám theo sau tối đa 2 lần dò (mặc định dò mỗi 3 frame, tức khoảng 0,4 giây ở 15 fps) miễn là camera còn thấy đủ 4 marker. Muốn bám nhanh hơn: `--aruco-every 2` hoặc `1` (tốn thêm CPU — đo bằng mục 7).
- Nếu ô lục/đỏ lệch khỏi marker thật trên ảnh thì vị trí bảng đang nhớ đã cũ: bỏ tay ra cho camera thấy lại đủ 4 marker.
- Khi đã thấy đủ 4 marker, node chỉ dò lại trong vùng ảnh quanh bảng (rẻ hơn dò cả khung); thiếu marker thì lần sau tự dò lại cả khung.
- Kiểm tra nhanh: chạm đầu bút vào dấu + → x, y và "cách mặt bảng" đều gần 0.

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
| `--calib` | File calib camera | `calib/c930e_720p.npz` (mặc định, calib lại 2026-10-07). Thiếu file này thì XYZ sai. Tự quy đổi cho 640×360 |
| `--trust-motion` | Độ bám của bộ lọc: lớn hơn thì bám nhanh hơn nhưng rung hơn | `1.0`; thấy trễ thì `2`–`4`, thấy rung thì `0.5` |
| `--no-filter` | Tắt bộ lọc, dùng thẳng kết quả từng frame | Chỉ để so sánh độ trễ |
| `--ping-host` | Máy cần đo ping (chỉ có trong JSON port 8081) | Bỏ trống = laptop đang SSH vào Pi |
| `--board` | Dò bảng vẽ 4 marker, phát toạ độ đầu bút so với bảng | Bật khi có bảng vẽ trong khung hình |
| `--aruco-every` | Dò bảng mỗi N frame | `3` (mặc định). Nhỏ hơn thì bám bảng nhanh hơn, tốn CPU hơn |
| `--pen-dims-mm` | Kích thước bút đo bằng thước: Tip→Tail, Tip→đường L-R, L↔R | Mặc định `64 44 23`. Khai sai thì khoảng cách sai đúng theo tỉ lệ đó |
| `--stream-width` | Bề ngang ảnh stream | `640` (mặc định) |
| `--run-tag` | Tên gắn vào file log của lượt chạy | Ví dụ `360p_bang` |
| `--duration` | Tự dừng sau N giây | Dùng khi so sánh các lượt chạy |
| `--no-log` / `--log-dir` | Tắt / đổi chỗ ghi log lượt chạy | Mặc định ghi vào `logs/` |

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
| Log báo `❌ KHÔNG tìm thấy file calib` | Chạy lại mục 2 để đưa `calib/c930e_720p.npz` lên Pi |
| XYZ đuổi theo bút chậm | Tăng `--trust-motion` (vd `3`), hoặc thử `--no-filter` để so |
| `Không mở được camera` | Kiểm tra camera đã cắm; tắt `usb_cam` nếu đang chạy; thử `--device /dev/video1`. Liệt kê camera: `for d in /sys/class/video4linux/video*; do echo "$d: $(cat $d/name)"; done` |
| `No such file or directory` với file `.py` hoặc model | Chưa đưa code lên Pi → chạy lại mục 2 |
| Lỗi khi nạp file `.tflite` | Dùng bản `.onnx` cùng tên |
| Không có dòng `PEN` nào | Xem `conf_max`: gần 0 → model không thấy bút (thử đổi góc/ánh sáng); nhỉnh hơn ngưỡng một chút → hạ `--conf` |
| Stream trình duyệt không lên | Kiểm tra Terminal 2 đang chạy và laptop đang nối wifi của Pi |
| `ros2: command not found` | `source /opt/ros/humble/setup.bash` |
| Pi chạy chậm bất thường | `htop` xem tiến trình nào đang chiếm CPU |

---

## 7. Log mỗi lượt chạy, kiểm sai số, so sánh cấu hình

### Log

Mỗi lần chạy `run_pi4_ros2.py` tự ghi một file `~/aeroscript/logs/run_<ngày>_<giờ>[_<tag>].csv`: mỗi frame một dòng gồm toạ độ bút (so với camera và so với bảng), 4 điểm khớp trên ảnh, vị trí bảng, marker đang thấy, độ trễ từng chặng, thời gian dò bảng, CPU, RAM, nhiệt độ. Dòng đầu file là cấu hình lượt chạy. Lúc thoát (Ctrl+C) node in đường dẫn file.

```bash
cd ~/aeroscript
python3 scripts/analyze_run.py --last              # lượt vừa chạy
python3 scripts/analyze_run.py --last 2            # 2 lượt gần nhất, so sánh cạnh nhau
python3 scripts/analyze_run.py logs/run_A.csv logs/run_B.csv
```

Kèm theo mỗi log là thư mục `run_…_frames/` chứa ảnh thô chụp mỗi 5 giây (`--snap-every`, 0 = tắt), để soi lại bút và bảng thật trên ảnh.

Chép log và ảnh về laptop (chạy trên laptop): `mkdir -p ~/new_rl_ros2/CoVip/logs/pi && scp -r "piros2@192.168.50.1:~/aeroscript/logs/*" ~/new_rl_ros2/CoVip/logs/pi/`

Với lượt có `--board`, `analyze_run.py` in thêm phần **Kiểm bảng**: tỉ lệ kích thước của bản in có đúng như khai không, và tiêu cự camera có khớp file calib không (cần camera thấy đủ 4 marker ít nhất vài giây, bảng nghiêng 25–40° so với hướng nhìn).

### Kiểm sai số bằng bảng

Chạy node với `--board`, chạm mũi bút vào **dấu +** và **4 góc vùng vẽ**, mỗi điểm **giữ yên 2–3 giây**, rồi Ctrl+C và chạy `analyze_run.py --last`. Phần "Bút đứng yên" liệt kê từng lần giữ yên:

- toạ độ đo được so với camera và so với bảng, độ rung;
- điểm chuẩn gần nhất và độ lệch (mm);
- cột **đo/thật**: khoảng cách bút bị đo lệch mấy lần, tính bằng cách kéo đầu bút dọc tia nhìn tới mặt bảng (đúng khi mũi bút thật sự đang chạm bảng). `1.00` là đúng; `1.50` là đo xa gấp rưỡi.

Nếu tỉ lệ đo/thật ổn định ở mọi điểm và khác 1, script in luôn bộ `--pen-dims-mm` cho khớp. Thử bộ số đó ngay trên log cũ, không cần chạy lại: `python3 scripts/analyze_run.py --last --pen-dims-mm A B C`.

### So sánh cấu hình (có/không dò bảng, 360p/720p)

```bash
cd ~/aeroscript
scripts/run_compare.sh                  # 4 lượt × 60 giây: 360p và 720p, không bảng và có bảng
scripts/run_compare.sh 60 360 360b      # dò bảng tốn thêm bao nhiêu (ở 360p)
scripts/run_compare.sh 60 360b 720b     # 360p so với 720p (đều có bảng)
```

Script chạy lần lượt từng cấu hình, mỗi lượt tự dừng, cuối cùng in bảng so sánh: fps, tỉ lệ bắt bút, thời gian giải mã ảnh, thời gian model, độ trễ camera→XYZ, thời gian và tần suất dò bảng, CPU cả máy và của node, RAM, nhiệt độ, độ rung khi bút đứng yên. So 2 lượt thì có thêm cột chênh lệch. Trong mỗi lượt hãy làm cùng một việc với bút để so sánh công bằng. `web_video_server` vẫn chạy ở cửa sổ khác như mọi lần; CPU của nó nằm trong dòng "CPU cả máy".

Đổi model/camera/tuỳ chọn thêm bằng biến môi trường: `MODEL=… DEVICE=… EXTRA="--pen-dims-mm 43 29.5 15.5" scripts/run_compare.sh`.

### Đo từng tiến trình

Muốn tách riêng CPU/RAM của `web_video_server`, mở thêm một terminal SSH trong lúc node đang chạy:

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
