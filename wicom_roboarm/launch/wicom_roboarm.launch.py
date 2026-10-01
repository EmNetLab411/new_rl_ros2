from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("wicom_roboarm")
    # servo_config:=servos_newarm.yaml để dùng cánh tay mới (5 servo)
    servo_config_arg = DeclareLaunchArgument("servo_config", default_value="servos.yaml")
    servo_yaml = PathJoinSubstitution(
        [os.path.join(pkg_share, "config"), LaunchConfiguration("servo_config")])

    unified = Node(
        package="wicom_roboarm",
        executable="wicom_roboarm_unified_node.py",
        name="wicom_roboarm_unified",
        output="screen",
        parameters=[
            servo_yaml,
        ],
        remappings=[
            ("joint_states", "/pca9685_servo/joint_states"),
            ("enable",       "/pca9685_servo/enable"),
            ("disable",      "/pca9685_servo/disable"),
            ("home",         "/pca9685_servo/home"),
            ("command",      "/pca9685_servo/command"),
            ("trajectory",   "/pca9685_servo/trajectory"),
            ("hardware_status", "/pca9685_servo/hardware_status"),
        ],
    )

    return LaunchDescription([servo_config_arg, unified])
