cmake_minimum_required(VERSION 3.14)

project(qdtsne
    VERSION 3.0.1
    DESCRIPTION "Quick and dirty t-SNE implementation in C++"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(qdtsne INTERFACE)
add_library(libscran::qdtsne ALIAS qdtsne)

target_include_directories(qdtsne INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/qdtsne>)
target_compile_features(qdtsne INTERFACE cxx_std_17)

# Dependencies
option(QDTSNE_FETCH_EXTERN "Automatically fetch qdtsne's external dependencies." ON)
if(QDTSNE_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(knncolle_knncolle 3.0.0 CONFIG REQUIRED)
    find_package(ltla_aarand 1.0.0 CONFIG REQUIRED)
    find_package(ltla_subpar 0.4.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.3 CONFIG REQUIRED)
endif()

target_link_libraries(qdtsne INTERFACE knncolle::knncolle ltla::aarand ltla::subpar ltla::sanisizer)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(QDTSNE_TESTS "Build qdtsne's test suite." ON)
else()
    option(QDTSNE_TESTS "Build qdtsne's test suite." OFF)
endif()

if(QDTSNE_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qdtsne)

install(TARGETS qdtsne
    EXPORT qdtsneTargets)

install(EXPORT qdtsneTargets
    FILE libscran_qdtsneTargets.cmake
    NAMESPACE libscran::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_qdtsne)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_qdtsneConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_qdtsne)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_qdtsneConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libscran_qdtsneConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_qdtsneConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_qdtsne)
