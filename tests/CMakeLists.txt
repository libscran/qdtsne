include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

add_executable(
    libtest 
    src/SPTree.cpp
    src/tsne.cpp
    src/gaussian.cpp
    src/symmetrize.cpp
    src/utils.cpp
)

target_link_libraries(
    libtest
    gtest_main
    knncolle
    qdtsne
)

# Add coverage.
option(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
else()
    set(DO_CODE_COVERAGE OFF)
endif()

macro(add_common_properties target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)

    if(DO_CODE_COVERAGE)
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

add_common_properties(libtest)

# Create targets to test parallelization.
macro(create_partest target)
    add_executable(
        ${target}
        src/tsne.cpp
        src/SPTree.cpp
    )

    target_link_libraries(
        ${target} 
        gtest_main
        qdtsne
    )
endmacro()

create_partest(cuspartest)
target_compile_definitions(cuspartest PRIVATE CUSTOM_PARALLEL_TEST=1)
add_common_properties(cuspartest)

find_package(OpenMP)
if (OpenMP_FOUND)
    create_partest(omptest)
    target_link_libraries(omptest OpenMP::OpenMP_CXX)
    add_common_properties(omptest)
endif()
