add_library(
    backbone STATIC
    vgg.cpp
)

set_target_properties(backbone PROPERTIES CXX_STANDARD 17)
target_include_directories(backbone PUBLIC ${PROJECT_SOURCE_DIR}/backbone)
target_link_libraries(backbone PUBLIC ${TORCH_LIBRARIES} Boost::headers)