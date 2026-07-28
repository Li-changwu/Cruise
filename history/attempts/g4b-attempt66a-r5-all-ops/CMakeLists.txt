cmake_minimum_required(VERSION 3.14)
project(G4bAttempt66aR5AllOps)

if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(EXISTS "/usr/local/Ascend/cann")
  set(ASCEND_HOME_PATH "/usr/local/Ascend/cann")
else()
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_executable(dataflow_full_decoder_smoke dataflow_full_decoder_smoke.cpp)
target_include_directories(dataflow_full_decoder_smoke PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
  "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc")
target_compile_options(dataflow_full_decoder_smoke PRIVATE
  -O2 -std=c++11 -ftrapv -fstack-protector-all -fPIC)
target_link_libraries(dataflow_full_decoder_smoke PRIVATE
  -Wl,--whole-archive
  "${ASCEND_HOME_PATH}/lib64/libgraph.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so"
  "${ASCEND_HOME_PATH}/lib64/libflow_graph.so"
  "${ASCEND_HOME_PATH}/lib64/libge_runner.so"
  "${ASCEND_HOME_PATH}/lib64/libdflow_runner.so"
  "${ASCEND_HOME_PATH}/lib64/libfmk_parser.so"
  "${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so"
  -Wl,--no-whole-archive)
