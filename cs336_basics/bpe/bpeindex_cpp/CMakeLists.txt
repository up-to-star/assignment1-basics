cmake_minimum_required(VERSION 3.20)

project(bpeindex_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Python_FIND_VIRTUALENV FIRST)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# 使用Python执行命令找到pybind11的CMake目录
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
)

# 检查命令是否成功执行
if(NOT pybind11_result EQUAL 0)
    message(FATAL_ERROR "Failed to locate pybind11 CMake directory")
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(bpeindex_cpp bpeindex.cpp bpeindex_pybind.cpp)

# 链接pybind11库
target_link_libraries(bpeindex_cpp PRIVATE pybind11::module)

# 添加编译定义以避免Python 3.12中的兼容性问题
target_compile_definitions(bpeindex_cpp PRIVATE -DPYBIND11_DETAILED_ERROR_MESSAGES=0)

set_target_properties(bpeindex_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
