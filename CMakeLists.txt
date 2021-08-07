cmake_minimum_required(VERSION 3.14)

project(qdtsne
    VERSION 1.0.0
    DESCRIPTION "Quick and dirty t-SNE implementation in C++"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(qdtsne INTERFACE)

target_include_directories(qdtsne INTERFACE include/)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
