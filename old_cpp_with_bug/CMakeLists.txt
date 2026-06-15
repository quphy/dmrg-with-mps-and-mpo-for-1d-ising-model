cmake_minimum_required(VERSION 3.10)
project(dmrg_ising)
add_executable(dmrg main.cpp dmrg.cpp mpo.cpp mps.cpp)
target_include_directories(dmrg PRIVATE include)

find_package(
        Armadillo
        8.500.0
        REQUIRED
)
if (${ARMADILLO_FOUND})
    message("Found Armadillo Library,")
    message("     ARMADILLO_VERSION: " ${ARMADILLO_VERSION_STRING})
    message("     ARMADILLO_LIBRARIES   : " ${ARMADILLO_LIBRARIES})
    message("     ARMADILLO_INCLUDE_DIRS: " ${ARMADILLO_INCLUDE_DIRS})

    include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(dmrg ${ARMADILLO_LIBRARIES})
endif ()


target_compile_features(dmrg PRIVATE cxx_std_17)