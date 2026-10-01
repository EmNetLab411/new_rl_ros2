#!/usr/bin/env python3
"""
Gazebo cho tay MỚI (newarm, 4 khớp, bút gắn cứng đồng trục J4).

Bản sao của visual_servoing_test.launch.py với: urdf/newarm/newarm.xacro,
world có bảng workspace in thật (marker 30mm, ±75mm), controllers_newarm.yaml.
Tay cũ vẫn chạy bằng visual_servoing_test.launch.py như trước.

    ros2 launch visual_servoing newarm_sim.launch.py
    ros2 launch visual_servoing newarm_sim.launch.py headless:=true
    ros2 launch visual_servoing newarm_sim.launch.py elbow_lower:=-1.5708 elbow_upper:=1.5708
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('visual_servoing').find('visual_servoing')

    # Set Gazebo resource path
    models_path = os.path.join(pkg_share, 'models')
    share_parent = os.path.dirname(pkg_share)

    gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if gz_resource_path:
        new_gz_resource_path = f"{gz_resource_path}:{models_path}:{share_parent}"
    else:
        new_gz_resource_path = f"{models_path}:{share_parent}"

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=new_gz_resource_path
    )

    set_gz_ip = SetEnvironmentVariable(
        name='GZ_IP',
        value='127.0.0.1'
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='true = chỉ chạy server Gazebo (không GUI)')
    # Cửa sổ servo khuỷu (rad). Mặc định [-30°,150°] = cách lắp khuyến nghị;
    # sừng lắp giữa thì truyền ±1.5708 (khi đó phải dời bảng ra ~31cm).
    elbow_lower_arg = DeclareLaunchArgument('elbow_lower', default_value='-0.523599')
    elbow_upper_arg = DeclareLaunchArgument('elbow_upper', default_value='2.617994')
    draw_pen_path_arg = DeclareLaunchArgument(
        'draw_pen_path', default_value='true',
        description='Vẽ vệt bút (/drawing/pen_position) trong Gazebo')

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_servoing"),
                    "urdf",
                    "newarm",
                    "newarm.xacro",
                ]
            ),
            " elbow_lower:=", LaunchConfiguration('elbow_lower'),
            " elbow_upper:=", LaunchConfiguration('elbow_upper'),
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='log',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([
                    FindPackageShare('visual_servoing'),
                    'worlds',
                    'visual_servoing_newarm.world'
                ]),
                PythonExpression(["' -r -s' if '", LaunchConfiguration('headless'), "' == 'true' else ' -r'"])
            ]
        }.items()
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'newarm',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    joint_state_broadcaster_spawner = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            )
        ]
    )

    arm_controller_spawner = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['arm_controller'],
                output='screen'
            )
        ]
    )

    vision_detector = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='visual_servoing',
                executable='vision_aruco_detector',
                name='vision_aruco_detector',
                output='log',
                parameters=[{
                    'image_topic': '/camera/image_raw',
                    'show_gui': False,
                    'use_sim_time': True,
                    'board_marker_offset_m': 0.075,
                    'board_marker_size_m': 0.030,
                }]
            )
        ]
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # /clock: sim cũ không bridge; cần cho các node dùng use_sim_time (newarm_sim_draw)
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    gazebo_drawing_visualizer = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='visual_servoing',
                executable='gazebo_drawing_visualizer',
                name='gazebo_drawing_visualizer',
                output='log',
                parameters=[{
                    'use_sim_time': True,
                    'spawn_shape_waypoints': False,
                    'draw_pen_path': ParameterValue(LaunchConfiguration('draw_pen_path'), value_type=bool),
                }]
            )
        ]
    )

    return LaunchDescription([
        headless_arg,
        elbow_lower_arg,
        elbow_upper_arg,
        draw_pen_path_arg,
        set_gz_resource_path,
        set_gz_ip,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        gz_bridge,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        vision_detector,
        gazebo_drawing_visualizer,
    ])
