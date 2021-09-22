add_executable(
    speedtest 
    speedtest.cpp
)

#find_package(OpenMP REQUIRED)

target_link_libraries(
    speedtest 
    qdtsne
#    OpenMP::OpenMP_CXX
)

target_compile_options(
    speedtest PUBLIC -O2
)

