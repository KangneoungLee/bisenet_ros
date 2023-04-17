import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
             package='bisenet_ros',
             namespace='',
             executable='bisenet_ros',
             name='bisenet_ros_node',
             output='screen',
             parameters=[os.path.join(get_package_share_directory("bisenet_ros"), 'params', 'bisenet_ros.yaml')],
            )
        ]
        )
        
