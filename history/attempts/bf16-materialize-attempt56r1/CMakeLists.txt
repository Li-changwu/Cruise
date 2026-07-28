add_ops_compile_options(
    OP_NAME Bf16Materialize
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
)

target_sources(op_host_aclnn PRIVATE bf16_materialize_def.cpp)
target_sources(optiling PRIVATE bf16_materialize_tiling.cpp)
target_sources(opsproto PRIVATE bf16_materialize_infershape.cpp)

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../op_kernel
    DESTINATION ${IMPL_INSTALL_DIR}/ascendc/bf16_materialize
)
