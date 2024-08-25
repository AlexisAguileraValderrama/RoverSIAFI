cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(my_gazebo_plugins)

find_package(gazebo REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_msgs REQUIRED)


include_directories(
  ${rclcpp_INCLUDE_DIRS}
  ${gazebo_INCLUDE_DIRS}
  ${gazebo_msgs_INCLUDE_DIRS}
)

add_library(SimplePlugin SHARED SimplePlugin.cpp)
target_link_libraries(SimplePlugin ${GAZEBO_LIBRARIES})

ament_target_dependencies(SimplePlugin
  rclcpp
  gazebo_ros
  gazebo_msgs
  std_msgs
)

# Additional necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Install the plugin to the specified directory
install(TARGETS SimplePlugin
  LIBRARY DESTINATION lib
)

# Add ament package support
ament_package()

