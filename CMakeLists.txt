cmake_minimum_required (VERSION 3.12)
project(hasc)

add_compile_options(
       # -Wall -std=c++17 -O3 -march=native -funroll-loops -ffast-math -fopt-info-vec -fno-strict-aliasing
       -Wall -std=c++17 -O3 -march=native -funroll-loops -ffast-math -fopt-info-vec -fno-trapping-math -fno-math-errno -fargument-noalias
       #
       # Optimized build with debug statements -> useful for performance investigation with intel-advisor
       # -Wall -std=c++17 -O3 -march=native -funroll-loops -ffast-math -fopt-info-vec -fno-trapping-math -fno-math-errno -fargument-noalias -g
       # -Wall -std=c++17 -O3 -march=native -funroll-loops -ffast-math -fno-trapping-math -fno-math-errno -fargument-noalias -g
)
add_compile_definitions(
        NDEBUG
)

# set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(vcl)

# =======================
# Scalar product examples
# =======================

add_executable(scalar_product_v0 scalar_product_v0.cc)

add_executable(scalar_product_v1 scalar_product_v1.cc)
target_link_libraries(scalar_product_v1 Threads::Threads)

add_executable(scalar_product_v2 scalar_product_v2.cc)
target_link_libraries(scalar_product_v2 Threads::Threads)

add_executable(scalar_product_v3 scalar_product_v3.cc)
target_link_libraries(scalar_product_v3 Threads::Threads)

add_executable(scalar_product_v4 scalar_product_v4.cc)
target_link_libraries(scalar_product_v4 Threads::Threads)

add_executable(scalar_product_v5 scalar_product_v5.cc)
target_link_libraries(scalar_product_v5 Threads::Threads)

# ===================
# Matrix computations
# ===================

add_executable(matmul_seq_v1 matmul_seq_v1.cc)

add_executable(matmul_seq_v2 matmul_seq_v2.cc)

add_executable(matvec_v1 matvec_v1.cc)

add_executable(matvec_v2 matvec_v2.cc)

add_executable(transpose_v1 transpose_v1.cc)

# ==============
# Other examples
# ==============

add_executable(pointer_chasing pointer_chasing.cc)
