cmake_minimum_required(VERSION 3.5)
project(ros_rl_inference)

# -----------------------------------------------------------------------
# Detect ROS version
# -----------------------------------------------------------------------
if(DEFINED ENV{ROS_VERSION})
  set(ROS_VERSION $ENV{ROS_VERSION})
else()
  message(WARNING "ROS_VERSION not set, assuming ROS 2")
  set(ROS_VERSION 2)
endif()

message(STATUS "Building ros_rl_inference for ROS ${ROS_VERSION}")

# -----------------------------------------------------------------------
# ROS 1 (catkin)
# -----------------------------------------------------------------------
if(ROS_VERSION EQUAL 1)
  find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    message_generation
  )

  add_message_files(
    FILES
      msg/InferenceInput.msg
      msg/InferenceOutput.msg
  )

  add_service_files(
    FILES
      srv/GetModelInfo.srv
  )

  generate_messages(
    DEPENDENCIES
      std_msgs
  )

  catkin_package(
    CATKIN_DEPENDS rospy std_msgs message_runtime
  )

  catkin_install_python(
    PROGRAMS
      nodes/inference_node_ros1
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(
    DIRECTORY src/ros_rl_inference/
    DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/ros_rl_inference
    FILES_MATCHING PATTERN "*.py"
  )

  install(
    DIRECTORY config/ launch/ros1/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

# -----------------------------------------------------------------------
# ROS 2 (ament_cmake)
# -----------------------------------------------------------------------
elseif(ROS_VERSION EQUAL 2)
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_python REQUIRED)
  find_package(rclpy REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(rosidl_default_generators REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/InferenceInput.msg"
    "msg/InferenceOutput.msg"
    "srv/GetModelInfo.srv"
    DEPENDENCIES std_msgs
  )

  ament_python_install_package(ros_rl_inference
    PACKAGE_DIR src/ros_rl_inference
  )

  install(
    PROGRAMS
      nodes/inference_node_ros2
    DESTINATION lib/${PROJECT_NAME}
  )

  install(
    DIRECTORY config/ launch/ros2/
    DESTINATION share/${PROJECT_NAME}
  )

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
  endif()

  ament_package()
endif()
