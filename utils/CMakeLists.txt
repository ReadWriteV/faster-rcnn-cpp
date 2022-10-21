add_library(
    utils STATIC
    utils.cpp
)

# set_target_properties(utils PROPERTIES CXX_STANDARD 17)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/utils)
target_link_libraries(utils PUBLIC ${TORCH_LIBRARIES})