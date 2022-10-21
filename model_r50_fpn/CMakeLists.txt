add_library(
    r50_fpn_model STATIC
    backbone.cpp
    neck.cpp
    anchor.cpp
    bbox.cpp
    loss.cpp
    rpn_head.cpp
    rcnn_head.cpp
    detector.cpp
    trainer.cpp
)

set_target_properties(r50_fpn_model PROPERTIES CXX_STANDARD 17)
target_include_directories(r50_fpn_model PUBLIC ${PROJECT_SOURCE_DIR}/model)
target_link_libraries(r50_fpn_model PUBLIC cvops utils dataset ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES} Boost::headers)


