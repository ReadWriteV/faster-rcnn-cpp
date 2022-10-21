cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(faster_rcnn)

enable_language(CUDA)
add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
add_definitions(-DWITH_CUDA)

set(CMAKE_PREFIX_PATH ../library)

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options headers)
find_package(OpenCV REQUIRED)

add_subdirectory(utils)
add_subdirectory(dataset)
add_subdirectory(cvops)
add_subdirectory(backbone)
add_subdirectory(model)
add_subdirectory(vgg_faster_rcnn_model)
add_subdirectory(eval)

add_executable(train train.cpp)
set_target_properties(train PROPERTIES CXX_STANDARD 17)
target_link_libraries(train PRIVATE model dataset Boost::program_options Boost::headers)

add_executable(test test.cpp)
set_target_properties(test PROPERTIES CXX_STANDARD 17)
target_link_libraries(test PRIVATE model dataset Boost::program_options Boost::headers)
