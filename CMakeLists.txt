cmake_minimum_required(VERSION 3.14)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 关键修改：添加正确的 include 路径
include_directories(${PROJECT_SOURCE_DIR}/lib)
include_directories(${PROJECT_SOURCE_DIR})

# 添加源文件
set(SOURCES
    src/common/logger.cpp
    src/nr_ldpc.cpp
)

# 添加可执行文件
add_executable(my_test_app test/main.cpp ${SOURCES})

# add_executable(my_test_app test/main.cpp)