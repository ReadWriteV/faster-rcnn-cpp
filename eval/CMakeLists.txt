add_library(eval STATIC eval.cpp)
set_target_properties(eval PROPERTIES CXX_STANDARD 17)
target_include_directories(eval PUBLIC ${PROJECT_SOURCE_DIR}/eval)
target_link_libraries(eval PUBLIC Boost::headers)

add_executable(test_eval test_eval.cpp)
set_target_properties(test_eval PROPERTIES CXX_STANDARD 17)
target_link_libraries(test_eval PUBLIC eval Boost::program_options)
