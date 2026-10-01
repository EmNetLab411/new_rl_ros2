#!/usr/bin/env python3
"""
Sinh mô tả Gazebo của tay mới (newarm) từ bản export Fusion 360 trong
ref/newarm_final_description-*/arm_ass_final_description:

    urdf/newarm/newarm.xacro                 chuỗi khớp đã sửa 2 lỗi export
    urdf/newarm/newarm.ros2_control.xacro    4 khớp position
    urdf/newarm/newarm.gazebo.xacro          thuộc tính Gazebo + cảm biến camera
    meshes/newarm/*.stl                      copy lưới (base_link (1).stl -> base_link.stl)
    models/newarm_board/                     bảng workspace in thật (marker 30mm, ±75mm)
    worlds/visual_servoing_newarm.world

Hai lỗi export được sửa (xem fk_newarm.py):
  1. Origin J1 lệch 0.42m -> lấy _J1_ORIGIN/_J2_ORIGIN của fk_newarm.
  2. Lưới STL export ở tư thế tay duỗi ngang (q2=+90°) trong khi chuỗi khớp là
     tay treo thẳng xuống -> origin visual/inertial của mọi link sau J2 được
     tính lại = nghịch đảo pose link ở tư thế CAD.

Bố cục world giữ quy ước của sim cũ: bảng ở phía +X world, mặt quay về -X,
trục J1 trùng trục Z world. base_link vì thế xoay yaw +90° ("phía trước" của
tay là -Y base_link).

Chạy lại script này mỗi khi đổi bản export hoặc đổi vị trí bảng:
    python3 newarm_make_sim.py
"""
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import fk_newarm as F

PKG = Path(__file__).resolve().parents[2]
REPO = PKG.parents[2]
REF = next((REPO / "ref").glob("newarm_final_description-*")) / "arm_ass_final_description"

# Khớp động: tên trong bản export -> tên dùng chung với driver thật
JOINT_RENAME = {"Revolute 2": "base", "Revolute 3": "shoulder",
                "Revolute 7": "elbow", "Revolute 24": "wrist_roll"}
ORIGIN_FIX = {"Revolute 2": F._J1_ORIGIN, "Revolute 3": F._J2_ORIGIN}
Q_CAD = {"shoulder": math.pi / 2}          # tư thế của lưới STL

# Giới hạn khớp trong sim (độ). Khuỷu theo cách lắp khuyến nghị [-30,150]
# (sừng servo lệch, home_deg=30); đổi bằng xacro arg elbow_lower/elbow_upper.
LIMITS_DEG = {"base": (-90, 90), "shoulder": (-90, 90),
              "elbow": (-30, 150), "wrist_roll": (-90, 90)}

# ── Bố cục world ─────────────────────────────────────────────────────────────
BASE_LIFT = 0.10            # nâng base_link để đầu bút (q=0) cách sàn ~11cm
BASE_YAW = math.pi / 2
BOARD_D = 0.225             # trục J1 -> mặt bảng (m)
BOARD_U = -0.015            # tâm bảng theo X base_link (m)
BOARD_BELOW_J1 = 0.163      # tâm bảng thấp hơn servo J1 (m)
MARKER_OFFSET = 0.075
MARKER_SIZE = 0.030
BOARD_SIZE = 0.190
# Camera cố định (eye-to-hand), đặt lệch bên + cao hơn để tay ít che marker
CAM_POS_BASE = (0.20, 0.26, None)   # x, y trong base_link; z = tâm bảng + 0.12


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _tf(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _axis_rot(axis, q):
    ax = np.array(axis, float)
    if abs(abs(ax[0]) - 1) < 1e-9:
        return _rx(q * ax[0])
    if abs(abs(ax[2]) - 1) < 1e-9:
        return _rz(q * ax[2])
    raise ValueError(f"trục khớp lạ: {axis}")


def _rpy(R):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (quy ước URDF)."""
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    return (math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0]))


def _f(v):
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _xyz(v):
    return " ".join(_f(x) for x in v)


def _vec(s):
    return [float(x) for x in s.split()]


# ── đọc bản export ───────────────────────────────────────────────────────────
def load_export():
    txt = (REF / "urdf" / "arm_ass_final.xacro").read_text()
    txt = re.sub(r"<xacro:include[^>]*/>", "", txt)
    root = ET.fromstring(txt)
    links, joints = {}, []
    for ln in root.findall("link"):
        ine, vis = ln.find("inertial"), ln.find("visual")
        I = ine.find("inertia").attrib
        links[ln.get("name")] = {
            "com": _vec(ine.find("origin").get("xyz")),
            "mass": float(ine.find("mass").get("value")),
            "inertia": {k: float(I[k]) for k in ("ixx", "iyy", "izz", "ixy", "iyz", "ixz")},
            "vis": _vec(vis.find("origin").get("xyz")),
            "mesh": Path(vis.find("geometry/mesh").get("filename")).name,
        }
    for j in root.findall("joint"):
        name = j.get("name")
        joints.append({
            "export_name": name,
            "name": JOINT_RENAME.get(name, name.lower().replace(" ", "_")),
            "type": j.get("type"),
            "xyz": list(ORIGIN_FIX.get(name, _vec(j.find("origin").get("xyz")))),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "axis": _vec(j.find("axis").get("xyz")) if j.find("axis") is not None else None,
        })
    return links, joints


def link_poses(joints, q):
    """Pose mỗi link trong base_link với q = {tên khớp: rad}."""
    T = {"base_link": np.eye(4)}
    pending = list(joints)
    while pending:
        for j in pending:
            if j["parent"] in T:
                R = _axis_rot(j["axis"], q.get(j["name"], 0.0)) if j["type"] == "revolute" else np.eye(3)
                T[j["child"]] = T[j["parent"]] @ _tf(p=j["xyz"]) @ _tf(R=R)
                pending.remove(j)
                break
        else:
            raise ValueError("cây khớp không liên thông")
    return T


# ── xacro ────────────────────────────────────────────────────────────────────
def make_xacro(links, joints, world_fix, cam):
    T_cad = link_poses(joints, Q_CAD)
    out = ['<?xml version="1.0" ?>',
           "<!-- SINH TỰ ĐỘNG bởi scripts/rl/newarm_make_sim.py — đừng sửa tay. -->",
           '<robot name="newarm" xmlns:xacro="http://www.ros.org/wiki/xacro">',
           "",
           f'<xacro:arg name="elbow_lower" default="{_f(math.radians(LIMITS_DEG["elbow"][0]))}"/>',
           f'<xacro:arg name="elbow_upper" default="{_f(math.radians(LIMITS_DEG["elbow"][1]))}"/>',
           '<xacro:include filename="$(find visual_servoing)/urdf/newarm/materials.xacro" />',
           '<xacro:include filename="$(find visual_servoing)/urdf/newarm/newarm.ros2_control.xacro" />',
           '<xacro:include filename="$(find visual_servoing)/urdf/newarm/newarm.gazebo.xacro" />',
           "",
           "<!-- base_link cố định vào world: trục J1 trùng trục Z world, tay hướng về +X -->",
           '<link name="world"/>',
           '<joint name="world_fixed" type="fixed">',
           f'  <origin xyz="{_xyz(world_fix[:3, 3])}" rpy="{_xyz(_rpy(world_fix[:3, :3]))}"/>',
           '  <parent link="world"/>',
           '  <child link="base_link"/>',
           "</joint>", ""]
    for name, L in links.items():
        Tinv = np.linalg.inv(T_cad[name])
        # toạ độ base_link (tư thế CAD) của trọng tâm: frame export = -vis
        com_base = np.array(L["com"]) - np.array(L["vis"])
        com = Tinv[:3, :3] @ com_base + Tinv[:3, 3]
        rpy = _xyz(_rpy(Tinv[:3, :3]))
        mass = max(L["mass"], 1e-3)
        I = {k: (max(v, 1e-8) if k in ("ixx", "iyy", "izz") else v) for k, v in L["inertia"].items()}
        mesh = "base_link.stl" if name == "base_link" else L["mesh"]
        out += [f'<link name="{name}">',
                "  <inertial>",
                f'    <origin xyz="{_xyz(com)}" rpy="{rpy}"/>',
                f'    <mass value="{mass:.6g}"/>',
                "    <inertia " + " ".join(f'{k}="{v:.3g}"' for k, v in I.items()) + "/>",
                "  </inertial>",
                "  <visual>",
                f'    <origin xyz="{_xyz(Tinv[:3, 3])}" rpy="{rpy}"/>',
                "    <geometry>",
                f'      <mesh filename="package://visual_servoing/meshes/newarm/{mesh}" scale="0.001 0.001 0.001"/>',
                "    </geometry>",
                f'    <material name="{"pen_red" if name == "but_1" else "silver"}"/>',
                "  </visual>",
                "</link>", ""]
    for j in joints:
        out += [f'<joint name="{j["name"]}" type="{j["type"]}">',
                f'  <origin xyz="{_xyz(j["xyz"])}" rpy="0 0 0"/>',
                f'  <parent link="{j["parent"]}"/>',
                f'  <child link="{j["child"]}"/>']
        if j["type"] == "revolute":
            lo, hi = (math.radians(v) for v in LIMITS_DEG[j["name"]])
            if j["name"] == "elbow":
                lim = 'lower="$(arg elbow_lower)" upper="$(arg elbow_upper)"'
            else:
                lim = f'lower="{_f(lo)}" upper="{_f(hi)}"'
            out += [f'  <axis xyz="{_xyz(j["axis"])}"/>',
                    f'  <limit {lim} effort="100" velocity="100"/>']
        out += ["</joint>", ""]
    out += [
        "<!-- Đầu bút = hopbut_1 + fk_newarm.TOOL_OFFSET (để so TF với FK) -->",
        '<link name="pen_tip"/>',
        '<joint name="pen_tip_fixed" type="fixed">',
        f'  <origin xyz="{_xyz(F.TOOL_OFFSET)}" rpy="0 0 0"/>',
        '  <parent link="hopbut_1"/>',
        '  <child link="pen_tip"/>',
        "</joint>", "",
        "<!-- Camera cố định (eye-to-hand), nhìn vào tâm bảng -->",
        '<link name="camera_link">',
        "  <inertial>",
        '    <mass value="0.01"/>',
        '    <inertia ixx="1e-05" iyy="1e-05" izz="1e-05" ixy="0" iyz="0" ixz="0"/>',
        "  </inertial>",
        "  <visual>",
        '    <geometry><box size="0.03 0.03 0.03"/></geometry>',
        '    <material name="camera_blue"/>',
        "  </visual>",
        "</link>",
        '<joint name="camera_joint" type="fixed">',
        f'  <origin xyz="{_xyz(cam[:3, 3])}" rpy="{_xyz(_rpy(cam[:3, :3]))}"/>',
        '  <parent link="base_link"/>',
        '  <child link="camera_link"/>',
        "</joint>",
        '<link name="camera_optical_link"/>',
        "<!-- Khung quang học: z ra trước, x sang phải, y xuống -->",
        '<joint name="camera_optical_joint" type="fixed">',
        '  <origin xyz="0 0 0" rpy="-1.5708 0 -1.5708"/>',
        '  <parent link="camera_link"/>',
        '  <child link="camera_optical_link"/>',
        "</joint>", "",
        "</robot>", ""]
    return "\n".join(out)


def make_ros2_control():
    out = ['<?xml version="1.0"?>',
           "<!-- SINH TỰ ĐỘNG bởi scripts/rl/newarm_make_sim.py -->",
           '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
           '  <ros2_control name="GazeboSystem" type="system">',
           "    <hardware>",
           "      <plugin>gz_ros2_control/GazeboSimSystem</plugin>",
           "    </hardware>"]
    for n in F.JOINT_NAMES:
        lo, hi = (math.radians(v) for v in LIMITS_DEG[n])
        if n == "elbow":
            lo_s, hi_s = "$(arg elbow_lower)", "$(arg elbow_upper)"
        else:
            lo_s, hi_s = _f(lo), _f(hi)
        out += [f'    <joint name="{n}">',
                '      <command_interface name="position">',
                f'        <param name="min">{lo_s}</param>',
                f'        <param name="max">{hi_s}</param>',
                "      </command_interface>",
                '      <state_interface name="position"/>',
                '      <state_interface name="velocity"/>',
                "    </joint>"]
    out += ["  </ros2_control>",
            "  <gazebo>",
            '    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">',
            "      <parameters>$(find visual_servoing)/config/controllers_newarm.yaml</parameters>",
            "      <ros>",
            "        <remapping>/controller_manager/robot_description:=/robot_description</remapping>",
            "      </ros>",
            "    </plugin>",
            "  </gazebo>",
            "</robot>", ""]
    return "\n".join(out)


def make_gazebo(links):
    out = ['<?xml version="1.0" ?>',
           "<!-- SINH TỰ ĐỘNG bởi scripts/rl/newarm_make_sim.py",
           "     Giống sim tay cũ: tắt trọng lực + tự va chạm (điều khiển vị trí thuần). -->",
           '<robot name="newarm" xmlns:xacro="http://www.ros.org/wiki/xacro">',
           '<xacro:macro name="link_gazebo" params="link_name">',
           '  <gazebo reference="${link_name}">',
           "    <self_collide>false</self_collide>",
           "    <gravity>false</gravity>",
           "  </gazebo>",
           "</xacro:macro>"]
    out += [f'<xacro:link_gazebo link_name="{n}"/>' for n in links]
    out += ["",
            '<gazebo reference="camera_link">',
            '  <sensor name="camera" type="camera">',
            "    <visualize>true</visualize>",
            "    <update_rate>30.0</update_rate>",
            "    <camera>",
            "      <horizontal_fov>1.3962634</horizontal_fov>",
            "      <image><width>640</width><height>480</height><format>R8G8B8</format></image>",
            "      <clip><near>0.02</near><far>300</far></clip>",
            "    </camera>",
            "    <topic>camera/image_raw</topic>",
            "  </sensor>",
            "</gazebo>",
            "</robot>", ""]
    return "\n".join(out)


MATERIALS = """<?xml version="1.0" ?>
<robot name="newarm" xmlns:xacro="http://www.ros.org/wiki/xacro" >
<material name="silver"><color rgba="0.700 0.700 0.700 1.000"/></material>
<material name="pen_red"><color rgba="0.800 0.100 0.100 1.000"/></material>
<material name="camera_blue"><color rgba="0 0 0.8 1"/></material>
</robot>
"""


# ── bảng + world ─────────────────────────────────────────────────────────────
def make_board_model():
    o, s = MARKER_OFFSET, MARKER_SIZE
    # Hệ model: bảng trong mặt YZ, mặt vẽ quay về -X. Nhìn từ phía tay: +Y bên trái.
    pos = {0: (o, o), 1: (-o, o), 2: (-o, -o), 3: (o, -o)}   # (y, z): 0 TL, 1 TR, 2 BR, 3 BL
    vis = []
    for i, (y, z) in pos.items():
        vis.append(f"""      <visual name="marker_{i}">
        <pose>-0.0012 {_f(y)} {_f(z)} 0 0 1.5708</pose>
        <geometry><box><size>{_f(s)} 0.002 {_f(s)}</size></box></geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <pbr><metal><albedo_map>model://aruco_marker_{i}/materials/textures/marker_{i}.png</albedo_map></metal></pbr>
        </material>
      </visual>""")
    h = 0.050
    frame = []
    for k, (y, z, sy, sz) in enumerate([(0, h, 0.1, 0.001), (0, -h, 0.1, 0.001),
                                        (h, 0, 0.001, 0.1), (-h, 0, 0.001, 0.1)]):
        frame.append(f"""      <visual name="draw_zone_{k}">
        <pose>-0.0006 {_f(y)} {_f(z)} 0 0 0</pose>
        <geometry><box><size>0.0002 {_f(sy)} {_f(sz)}</size></box></geometry>
        <material><ambient>0.6 0.6 0.6 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse></material>
      </visual>""")
    sdf = f"""<?xml version="1.0"?>
<!-- SINH TỰ ĐỘNG bởi scripts/rl/newarm_make_sim.py
     Bảng workspace in thật của tay mới: marker {s*1000:.0f}mm, tâm marker ±{o*1000:.0f}mm,
     vùng vẽ 100x100mm (khung xám). Không có collision: bút đi xuyên qua bảng. -->
<sdf version="1.8">
  <model name="newarm_board">
    <static>true</static>
    <link name="board_link">
      <visual name="paper">
        <geometry><box><size>0.001 {_f(BOARD_SIZE)} {_f(BOARD_SIZE)}</size></box></geometry>
        <material>
          <ambient>0.98 0.98 0.98 1</ambient>
          <diffuse>0.98 0.98 0.98 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>
{chr(10).join(vis)}
{chr(10).join(frame)}
    </link>
  </model>
</sdf>
"""
    cfg = """<?xml version="1.0"?>
<model>
  <name>newarm_board</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>Bảng workspace 4 ArUco (DICT_4X4_1000 ID 0-3) cho tay mới</description>
</model>
"""
    return sdf, cfg


def make_world(board_world):
    src = (PKG / "worlds" / "visual_servoing_training.world").read_text()
    a = src.index("    <!-- 4 ArUco Markers on drawing surface")
    b = src.index("    <!-- World coordinate frame visualization -->")
    board = f"""    <!-- Bảng workspace tay mới: mặt bảng (0.5mm trước tâm tấm giấy) cách trục J1
         {BOARD_D*100:.1f}cm, quay về -X. SINH bởi scripts/rl/newarm_make_sim.py -->
    <include>
      <uri>model://newarm_board</uri>
      <name>newarm_board</name>
      <pose>{_f(board_world[0] + 0.0005)} {_f(board_world[1])} {_f(board_world[2])} 0 0 0</pose>
    </include>

"""
    plugins = """    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

"""
    out = src[:a] + board + src[b:]
    out = out.replace("    <!-- Physics settings -->", plugins + "    <!-- Physics settings -->")
    out = out.replace("<pose>0.8 -0.3 0.4 0 0.3 2.0</pose>", "<pose>0.55 -0.55 0.65 0 0.45 2.2</pose>")
    return out


def main():
    links, joints = load_export()

    # world -> base_link: yaw 90°, trục J1 về (0,0), nâng BASE_LIFT
    R = _rz(BASE_YAW)
    j1 = np.array(F._J1_WORLD)
    t = -(R @ np.array([j1[0], j1[1], 0.0])) + np.array([0, 0, BASE_LIFT])
    world_fix = _tf(R, t)

    board_base = np.array([BOARD_U, j1[1] - BOARD_D, j1[2] - BOARD_BELOW_J1])
    board_world = R @ board_base + t

    cam_pos = np.array([CAM_POS_BASE[0], CAM_POS_BASE[1], board_base[2] + 0.12])
    fwd = board_base - cam_pos
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0, 0, 1], fwd)
    left /= np.linalg.norm(left)
    cam = _tf(np.column_stack([fwd, left, np.cross(fwd, left)]), cam_pos)

    urdf_dir = PKG / "urdf" / "newarm"
    mesh_dir = PKG / "meshes" / "newarm"
    board_dir = PKG / "models" / "newarm_board"
    for d in (urdf_dir, mesh_dir, board_dir):
        d.mkdir(parents=True, exist_ok=True)

    (urdf_dir / "newarm.xacro").write_text(make_xacro(links, joints, world_fix, cam))
    (urdf_dir / "newarm.ros2_control.xacro").write_text(make_ros2_control())
    (urdf_dir / "newarm.gazebo.xacro").write_text(make_gazebo(list(links)))
    (urdf_dir / "materials.xacro").write_text(MATERIALS)
    for stl in (REF / "meshes").glob("*.stl"):
        shutil.copyfile(stl, mesh_dir / ("base_link.stl" if stl.name.startswith("base_link") else stl.name))
    sdf, cfg = make_board_model()
    (board_dir / "model.sdf").write_text(sdf)
    (board_dir / "model.config").write_text(cfg)
    (PKG / "worlds" / "visual_servoing_newarm.world").write_text(make_world(board_world))

    # tự kiểm: chuỗi sinh ra phải khớp fk_newarm
    worst = 0.0
    for q in ([0, 0, 0, 0], [0.4, -0.7, 1.3, 0.5], [-1.2, 0.9, -0.3, -1.0]):
        T = link_poses(joints, dict(zip(F.JOINT_NAMES, q)))["hopbut_1"]
        worst = max(worst, float(np.abs(T - np.array(F.fk_flange_matrix(q))).max()))
    print(f"chuỗi xacro vs fk_newarm: lệch lớn nhất {worst:.2e}")
    print(f"world->base_link xyz={np.round(t, 6)} yaw=90°")
    print(f"tâm bảng: base_link {np.round(board_base, 4)}  world {np.round(board_world, 4)}")
    print(f"camera:   base_link {np.round(cam_pos, 4)}  world {np.round(R @ cam_pos + t, 4)}")


if __name__ == "__main__":
    main()
