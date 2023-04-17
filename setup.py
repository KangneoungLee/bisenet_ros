import os
from setuptools import setup

package_name = 'bisenet_ros'
submodule1 ='bisenet_ros/lib'
submodule2 ='bisenet_ros/nets'

share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodule1, submodule2],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join(share_dir, "launch"), ["launch/bisenet_ros.launch.py"]),
        (os.path.join(share_dir, "params"), ["params/bisenet_ros.yaml"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artlab',
    maintainer_email='eehoa12381@gmai.com',
    description='Bisenet ROS2 wrapper',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['bisenet_ros = bisenet_ros.bisenet_ros_node:main'
        ],
    },
)
