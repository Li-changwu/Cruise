cmake_minimum_required(VERSION 3.14)
project(Attempt74ResidentEpochBridge)

if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(EXISTS "/usr/local/Ascend/cann")
  set(ASCEND_HOME_PATH "/usr/local/Ascend/cann")
else()
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CANN_LIBRARIES
  "${ASCEND_HOME_PATH}/lib64/libgraph.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so"
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so"
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so"
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so"
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so"
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so")

add_library(resident_epoch_bridge_object OBJECT resident_epoch_bridge.cpp)
set_target_properties(resident_epoch_bridge_object PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(resident_epoch_bridge_object PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
  "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc")
target_compile_options(resident_epoch_bridge_object PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)

add_library(resident_epoch_bridge_old_object OBJECT resident_epoch_bridge_old.cpp)
set_target_properties(resident_epoch_bridge_old_object PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(resident_epoch_bridge_old_object PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
  "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc")
target_compile_options(resident_epoch_bridge_old_object PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)

add_library(resident_epoch_bridge SHARED
  $<TARGET_OBJECTS:resident_epoch_bridge_object>)
target_link_libraries(resident_epoch_bridge PRIVATE
  -Wl,--whole-archive
  ${CANN_LIBRARIES}
  -Wl,--no-whole-archive)

add_library(resident_epoch_bridge_old SHARED
  $<TARGET_OBJECTS:resident_epoch_bridge_old_object>)
target_link_libraries(resident_epoch_bridge_old PRIVATE
  -Wl,--whole-archive
  ${CANN_LIBRARIES}
  -Wl,--no-whole-archive)

add_library(resident_epoch_memcpy_trace SHARED rt_memcpy_trace.cpp)
target_include_directories(resident_epoch_memcpy_trace PRIVATE
  "${ASCEND_HOME_PATH}/include")
target_compile_options(resident_epoch_memcpy_trace PRIVATE
  -O2 -std=c++11 -fstack-protector-all -fPIC)
target_link_libraries(resident_epoch_memcpy_trace PRIVATE
  dl
  "${ASCEND_HOME_PATH}/lib64/libgraph.so")

add_executable(resident_epoch_server
  resident_epoch_server.cpp
  $<TARGET_OBJECTS:resident_epoch_bridge_object>)
target_include_directories(resident_epoch_server PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(resident_epoch_server PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)
target_link_libraries(resident_epoch_server PRIVATE
  -Wl,--whole-archive
  ${CANN_LIBRARIES}
  -Wl,--no-whole-archive)

add_executable(resident_epoch_server_old
  resident_epoch_server.cpp
  $<TARGET_OBJECTS:resident_epoch_bridge_old_object>)
target_include_directories(resident_epoch_server_old PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(resident_epoch_server_old PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)
target_link_libraries(resident_epoch_server_old PRIVATE
  -Wl,--whole-archive
  ${CANN_LIBRARIES}
  -Wl,--no-whole-archive)

add_executable(relocate_air_paths relocate_air_paths.cpp)
target_include_directories(relocate_air_paths PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external")
target_compile_options(relocate_air_paths PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)
target_link_libraries(relocate_air_paths PRIVATE
  -Wl,--whole-archive
  "${ASCEND_HOME_PATH}/lib64/libgraph.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so"
  -Wl,--no-whole-archive)
