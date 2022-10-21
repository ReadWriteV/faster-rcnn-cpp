add_library(
    dataset STATIC
    voc.cpp
    coco.cpp
)

set_target_properties(dataset PROPERTIES CXX_STANDARD 17)
target_include_directories(dataset PUBLIC ${PROJECT_SOURCE_DIR}/dataset)
target_link_libraries(dataset PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES} Boost::headers PRIVATE utils)
