cmake_minimum_required(VERSION 3.16)

project(ttl_cnn_schedule_search_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_executable(ttl_schedule_search_demo
  src/main.cpp
  src/schedule_search.cpp
)

target_include_directories(ttl_schedule_search_demo PRIVATE include)
target_link_libraries(ttl_schedule_search_demo PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(ttl_schedule_search_demo PRIVATE /W4)
else()
  target_compile_options(ttl_schedule_search_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
