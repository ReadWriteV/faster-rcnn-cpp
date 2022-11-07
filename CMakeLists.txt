cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(faster_rcnn)

enable_language(CUDA)
add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
add_definitions(-DWITH_CUDA)

set(CMAKE_PREFIX_PATH ../library)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options headers)
find_package(OpenCV REQUIRED)

add_subdirectory(utils)
add_subdirectory(dataset)
add_subdirectory(model_r50_fpn)
# add_subdirectory(model_vgg)
add_subdirectory(eval)

# add_executable(train train.cpp)
# set_target_properties(train PROPERTIES CXX_STANDARD 17)
# target_link_libraries(train PRIVATE model_vgg dataset Boost::program_options Boost::headers)

# add_executable(test test.cpp)
# set_target_properties(test PROPERTIES CXX_STANDARD 17)
# target_link_libraries(test PRIVATE model_vgg dataset Boost::program_options Boost::headers)

add_executable(train_r50fpn train_r50fpn.cpp)
set_target_properties(train_r50fpn PROPERTIES CXX_STANDARD 17)
target_link_libraries(train_r50fpn PRIVATE model_r50_fpn Boost::program_options Boost::headers)
