set(TARGET_NAME model_r50_fpn)

add_library(
    ${TARGET_NAME} STATIC
    resnet.cpp
    neck.cpp
    anchor.cpp
    bbox.cpp
    loss.cpp
    rpn_head.cpp
    rcnn_head.cpp
    detector.cpp
    trainer.cpp
)

set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC utils dataset ${TORCH_LIBRARIES} TorchVision::TorchVision ${OpenCV_LIBRARIES} Boost::headers)


