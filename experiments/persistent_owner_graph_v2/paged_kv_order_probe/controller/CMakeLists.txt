cmake_minimum_required(VERSION 3.14)
project(CruiseV2PagedKvOrderController)

if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(EXISTS "/usr/local/Ascend/cann")
  set(ASCEND_HOME_PATH "/usr/local/Ascend/cann")
else()
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN})
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_library(${UDF_TARGET_LIB} SHARED paged_kv_order_controller.cpp)
target_include_directories(${UDF_TARGET_LIB} PRIVATE ${ASCEND_HOME_PATH}/include)
target_compile_options(${UDF_TARGET_LIB} PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)

if("x${RESOURCE_TYPE}" STREQUAL "xAscend")
  target_link_libraries(${UDF_TARGET_LIB} PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/devlib/device/libflow_func.so
    ${ASCEND_HOME_PATH}/devlib/device/libmmpa.so
    -Wl,--no-whole-archive)
elseif("x${RESOURCE_TYPE}" STREQUAL "xAarch")
  target_link_libraries(${UDF_TARGET_LIB} PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/devlib/linux/aarch64/libflow_func.so
    -Wl,--no-whole-archive)
else()
  message(FATAL_ERROR "V2 Paged-KV order controller requires AArch or Ascend")
endif()
