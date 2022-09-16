cmake_minimum_required(VERSION 3.0.0)
project(RelComp VERSION 0.1.0)
#set(CMAKE_C_COMPILER "gcc-7")
#set(CMAKE_CXX_COMPILER "g++-7")


# include(CTest)
# enable_testing()

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)
link_libraries(stdc++fs)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
find_package(Boost)

add_executable(RelComp ${SRC_FILES})
target_compile_options(RelComp PUBLIC -std=c++11 -Wall)

target_link_libraries(RelComp PUBLIC ${Boost_LIBRARIES} Threads::Threads)

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
