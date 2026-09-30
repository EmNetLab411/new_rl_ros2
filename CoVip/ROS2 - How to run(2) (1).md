# ROS2 - How to run

**RUN 6-DOF:** Khởi chaỵ

```
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

Set Home - điều khiển từng góc

```
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[90.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['shoulder'], position:[180.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['elbow'], position:[0.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_roll'], position:[90.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_pitch'], position:[90.0]}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['pen'], position:[90.0]}"
```



 **Auto Draw square**

```
python3 ~/ros2_ws/src/wicom_roboarm/src/wicom_roboarm_4dof_standalone.py --ros-args \
  -p ch_base:=0 \
  -p ch_shoulder:=1 -p ch_shoulder_mirror:=2 -p shoulder_mirror_enabled:=true -p shoulder_mirror_angle_max:=180.0 \
  -p ch_elbow:=3 \
  -p ch_wrist_pitch:=5 \
  -p sign_shoulder:=-1.0 \
  -p offset_shoulder_deg:=30.0 \
  -p offset_wrist_pitch_deg:=30.0 \
  -p offset_elbow_deg:=-30.0 \
  -p fixed_channels:="[4,6]" -p fixed_degs:="[100.0,30.0]" \
  -p sign_elbow:=-1.0 \
  -p auto_draw:=true -p auto_loop:=true
```



**Stream Video:**

terminal 1:

```
ros2 run web_video_server web_video_server
```

teminal 2:

```
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:="/dev/video0" -p image_width:=640 -p image_height:=480 -p pixel_format:="yuyv"
```

Xem video: 

Laptop kết nối vào wifi của Pi mở chrome

```
http://192.168.50.1:8080/
```



Chạy Mavlink Router sau khi reboot px4

Bash

```
sudo systemctl restart mavlink-router
```

------

### 



















#### Cách 1: Biến Pi thành Web Server (Xem trên trình duyệt Chrome/Edge) - CỰC HAY

Bạn cài một gói nhỏ trên Pi, nó sẽ phát video ra dạng trang web. Bạn ngồi ở máy tính (Windows hay WSL đều được) mở Chrome lên là xem được.

**Bước 1: Trên Pi (ROS 2), cài đặt gói web video:**

Bash

```
sudo apt install ros-humble-web-video-server
```

**Bước 2: Chạy server:**

Bash

```
ros2 run web_video_server web_video_server
```

**Bước 3: Trên PC (Windows/WSL):** Mở trình duyệt web và gõ địa chỉ: `http://<IP_CUA_PI>:8080`

Bạn sẽ thấy một danh sách topic, bấm vào dòng `/image_raw` là xem được video trực tiếp, độ trễ cực thấp. Cách này tiện nhất vì không cần cài gì trên máy tính cả.



```
ros2 run camera_ros camera_node --ros-args -p width:=640 -p height:=480 -p format:="BGR888"

ros2 run rqt_image_view rqt_image_view
```



**4DOF**

ros2 launch wicom_roboarm wicom_roboarm_drawing_square.launch.py

```
python3 ~/ros2_ws/src/wicom_roboarm/src/wicom_roboarm_4dof_standalone.py --ros-args \
  -p ch_base:=0 -p ch_shoulder:=1 -p ch_elbow:=2 -p ch_wrist_pitch:=3 \
  -p sign_shoulder:=-1.0 \
  -p sign_elbow:=-1.0 -p sign_wrist:=-1.0 \
  -p offset_shoulder_deg:=30.0 \
  -p auto_draw:=true -p auto_loop:=true
```



## Fix 

**Run ros2 connect to PX4**

```
MicroXRCEAgent serial --dev /dev/ttyS0 -b 921600
```

**Tạo Service để tự chạy khi khởi động** Tạo file: `sudo nano /etc/systemd/system/uxrce_agent.service`

Ini, TOML

```
[Unit]
Description=MicroXRCE-DDS Agent
After=network.target

[Service]
ExecStart=/usr/local/bin/MicroXRCEAgent serial --dev /dev/ttyS0 -b 921600
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
```

Lưu lại, sau đó:

Bash

```
sudo systemctl enable uxrce_agent
sudo systemctl start uxrce_agent
```

------

Chạy Mavlink Router sau khi reboot px4

Bash

```
sudo systemctl restart mavlink-router
```

------

### Run Robotic arm 4DOF

```
python3 ~/ros2_ws/src/wicom_roboarm/src/wicom_roboarm_4dof_standalone.py --ros-args \
  -p i2c_bus:=1 -p pca_address:=0x40 -p use_mux:=true -p mux_address:=0x70 -p mux_channel:=2 \
  -p ch_base:=0 -p ch_shoulder:=1 -p ch_elbow:=2 -p ch_wrist_pitch:=3 \
  -p auto_draw:=true -p auto_loop:=true
```



### Run Offboard control

```
cd ros2_px4_teleop_example/
source install/setup.bash
ros2 run teleop teleop
```

```
source install/setup.bash
ros2 run teleop_twist_rpyt_keyboard teleop_twist_rpyt_keyboard
```



```
cd ros2_ws/
source install/setup.bash
ros2 run px4_ros_com offboard_control
```



# RUN ROS2 SIMULATION

```
PX4_GZ_WORLD=walls make px4_sitl gz_x500
```

```
MicroXRCEAgent udp4 -p 8888
```

```
cd ~/ws_ros2
source install/local_setup.bash
ros2 run px4_ros_com offboard_control
```



Bạn hãy tìm (gõ vào ô Search) và sửa 2 thông số sau:

1. **`NAV_DLL_ACT`** (Navigation Data Link Loss Action - Hành động khi mất kết nối GCS)
   - Mặc định: `Return` (hoặc `Hold`).
   - **Sửa thành:** **`Disabled`** (Vô hiệu hóa) hoặc **`0`**.
   - *Giải thích: Bảo PX4 là "Nếu mất kết nối với máy tính QGC thì kệ nó, cứ bay tiếp theo lệnh của ROS 2 đi".*
2. **`COM_DL_LOSS_T`** (Data Link Loss Timeout - Thời gian chờ trước khi báo mất kết nối)
   - Mặc định: `10` (giây).
   - **Sửa thành:** **`60`** hoặc cao hơn.
   - *Giải thích: Tăng thời gian chờ lên để nếu mạng WSL có bị lag một chút thì Drone cũng không hoảng loạn mà hạ cánh ngay.*



```
dos2unix ~/ros2_ws/src/wicom_roboarm/src/wicom_roboarm_drawing_ik_node.py
cd ~/ros2_ws
colcon build --symlink-install --packages-select wicom_roboarm
source install/setup.bash

ros2 launch wicom_roboarm wicom_roboarm.launch.py
# terminal khác:
ros2 run wicom_roboarm wicom_roboarm_drawing_ik_node.py
# terminal khác:
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 20.0, y: 0.0, z: 15.0}"
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['shoulder'], position:[90.0]}"
```


#Detect Pen_tip: Xây dựng phương pháp đo lường qua ảnh cho Pi4 sử dụng giải thuật PnP
(in Pi4)
- terminal 1: ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:="/dev/video0" -p image_width:=640 -p image_height:=480 -p pixel_format:="yuyv" -p framerate:=15.0
	      ### ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0 -p image_width:=640 -p image_height:=480 -p frame_rate:=15
	      ### ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:="/dev/video0" -p image_width:=320 -p image_height:=240 -p pixel_format:="yuyv" -p framerate:=15.0
- terminal 2: ros2 run web_video_server web_video_server
- terminal 3: cd aeroscript
		python3 run_pi4_ros2.py --threads 1 --trust-motion 0.5
		### python3 run_pi4_ros.py --model best_int8.onnx
		### python3 run_pi4_ros.py --model best.onnx (highest accuracy)
		###python3 run_pi4_ros.py --model best.onnx --target-fps 0 --threads 1
		### python3 run_pi4_ros.py --model best_int8.onnx --target-fps 12
(in laptop)				
- terminal 4: cd ComputerVisionCalib 
		python3 view_stream.py
		###python3 view_stream.py --piros2 192.168.50.1 --display-fps 15
- Check cpu: htop
- LiDAR
ls /dev/ttyUSB*
sudo chmod 777 /dev/ttyUSB0
cd ldlidar_ros2_ws/src
cd ldlidar_stl_ros2
cd launch
gedit ld06.launch.py
cd ..
cd ..
cd ..
colcon build (ldlidar_ros2_ws)
source install/setup.bash
source ~/.bashrc
ros2 launch ldlidar_stl_ros2 ld06.launch.py
terminal 2: ros2 topic list
		rivz
fixed frame -> base link
add -> laserscan -> topic -> /scan
add -> TF
