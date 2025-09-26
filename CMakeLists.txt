cmake_minimum_required(VERSION 3.0.2)
project(pillar_detect)

# 设置 C++14 标准（ROS Noetic 推荐，且避免 macOS C++17 兼容问题）
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Catkin 和所需 ROS 包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs      # 必须！PointCloud2 在此包中
  geometry_msgs
  pcl_ros          # 推荐显式声明，用于 fromROSMsg/toROSMsg
  tf               # 添加 TF 支持
)

# 查找 PCL
find_package(PCL REQUIRED)

# 声明包信息（供其他包依赖时使用）
catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    pcl_ros
    tf
)

# 包含头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# 声明可执行文件
add_executable(pillar_detect src/pillar_detect.cpp)

# 添加依赖：确保自动生成的消息头文件先被构建
add_dependencies(pillar_detect
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# 链接库
target_link_libraries(pillar_detect
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 安装目标
install(TARGETS pillar_detect
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装launch文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# 安装RViz配置
install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)