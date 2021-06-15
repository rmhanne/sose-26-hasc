cmake_minimum_required (VERSION 3.12)
project(hasc)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============
# Compiler flags
# ==============

# Notes:
# - icc will not be able to build everything!
# - dpcpp also has some problems, eg with code that uses eigen

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
  add_compile_options(-Wall -O3 -march=native -fno-trapping-math -funroll-loops -ffast-math -fargument-noalias)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-fabi-version=0)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
endif ()

add_compile_definitions(
        NDEBUG
)

# =================
# Find dependencies
# =================

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP)

find_package(MPI)

find_package (TBB)

# The vectorclass library should be checked out in a subfolder called vcl. If
# you don't have it you probably didn't clone with `git clone --recursive ...`
include_directories(vcl)

# It seems as if there is no official find_package for the MKL library. There
# have been some user written scripts but apparently most of them are outdated.
# Intel itself doesn't seem to be active about doing anything. So I can't
# properly guard the matmul_libraries target.
#
# This means: Make sure you have intel-mkl installed and that all the paths are
# set properly. If you installed intel using their installer it should have
# pointed you to a `setvars.sh` script. Call this in the shell where you want to
# build the executable and it should be able to find mkl. In my case this is:
#
# . ~/intel/oneapi/setvars.sh
#
# As a work around we use an option. You can set the option to `ON` if you have
# intel-mkl installed and want to build the targets. Note: You can also call cmake with
#
# cmake -DHASC_HAVE_MKL /path/to/source/dir
#
# You might need to remove the CMake cache before.

option(HASC_HAVE_MKL "Intel-mkl is installed and all path variables are set so it can be found. Work around not having a find_package(MKL)." OFF)

# Similar issue with SYCL

option(HASC_HAVE_SYCL "You use a compiler that knows how to compile code with sycl statements." OFF)

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

add_executable(scalar_product_ms scalar_product_ms.cc)
target_link_libraries(scalar_product_ms Threads::Threads)

# ==================
# Matrix computation
# ==================

add_executable(matmul_seq_v1 matmul_seq_v1.cc)

add_executable(matmul_seq_v2 matmul_seq_v2.cc)

if (OpenMP_FOUND)
  add_executable(matmul_omp matmul_omp.cc)
  target_link_libraries(matmul_omp PRIVATE OpenMP::OpenMP_CXX)
endif (OpenMP_FOUND)

if (OpenMP_FOUND)
  add_executable(matmul_omp_avx512 matmul_omp_avx512.cc)
  target_link_libraries(matmul_omp_avx512 PRIVATE OpenMP::OpenMP_CXX)
endif (OpenMP_FOUND)

add_executable(transpose_v1 transpose_v1.cc)

add_executable(matvec_v1 matvec_v1.cc)

add_executable(matvec_v2 matvec_v2.cc)

add_executable(transpose_vectorized transpose_vectorized.cc)

if (HASC_HAVE_MKL)
find_package (Eigen3 3.3 NO_MODULE)
if (TARGET Eigen3::Eigen)
  add_executable(matmul_libraries matmul_libraries.cc)
  target_link_libraries (matmul_libraries Eigen3::Eigen)
  target_link_libraries(matmul_libraries mkl_intel_lp64 mkl_sequential mkl_core)

  add_executable(transpose_libraries transpose_libraries.cc)
  target_link_libraries (transpose_libraries Eigen3::Eigen)
  target_link_libraries(transpose_libraries mkl_intel_lp64 mkl_sequential mkl_core)
endif (TARGET Eigen3::Eigen)
endif (HASC_HAVE_MKL)


# =============
# Nbody Example
# =============

add_executable(nbody_vanilla nbody_vanilla.cc)

add_executable(nbody_vectorized nbody_vectorized.cc)

if (MPI_FOUND)
  add_executable(nbody_mpi nbody_mpi.cc)
  target_link_libraries(nbody_mpi PRIVATE MPI::MPI_CXX)
endif (MPI_FOUND)

if (OpenMP_FOUND)
  add_executable(nbody_omp nbody_omp.cc)
  target_link_libraries(nbody_omp PRIVATE OpenMP::OpenMP_CXX)
endif (OpenMP_FOUND)

add_executable(nbody_vectorized_threaded nbody_vectorized_threaded.cc)
target_link_libraries(nbody_vectorized_threaded Threads::Threads)

add_executable(nbody_vectorized_threaded_v2 nbody_vectorized_threaded_v2.cc)
target_link_libraries(nbody_vectorized_threaded_v2 Threads::Threads)

if (TBB_FOUND)
  add_executable(nbody_tbb nbody_tbb.cc)
  target_link_libraries(nbody_tbb Threads::Threads)
  target_link_libraries(nbody_tbb TBB::tbb)

  add_executable(nbody_tbb_v2 nbody_tbb_v2.cc)
  target_link_libraries(nbody_tbb_v2 TBB::tbb)
endif (TBB_FOUND)

# This one uses atomic updates for data type double and needs C++-20
add_executable(nbody_threads nbody_threads.cc)
target_link_libraries(nbody_threads Threads::Threads)
set_property(TARGET nbody_threads PROPERTY CXX_STANDARD 20)

if (HASC_HAVE_SYCL)
  add_executable(nbody_sycl nbody_sycl.cc)
endif (HASC_HAVE_SYCL)


# ==============
# Other examples
# ==============

add_executable(pointer_chasing pointer_chasing.cc)

# ===========
# C++ Threads
# ===========

add_executable(peterson peterson.cc)
target_link_libraries(peterson Threads::Threads)

add_executable(philosophers philosophers.cc)
target_link_libraries(philosophers Threads::Threads)

add_executable(barrier barrier.cc)
target_link_libraries(barrier Threads::Threads)

add_executable(power_method power_method.cc)
target_link_libraries(power_method Threads::Threads)

add_executable(packaged_task packaged_task.cc)
target_link_libraries(packaged_task Threads::Threads)

add_executable(producer_consumer producer_consumer.cc)
target_link_libraries(producer_consumer Threads::Threads)

# ======
# OpenMP
# ======

if (OpenMP_FOUND)
  add_executable(hello_openmp hello_openmp.cc)
  target_link_libraries(hello_openmp PRIVATE OpenMP::OpenMP_CXX)
endif (OpenMP_FOUND)

# ===
# TBB
# ===

if (TBB_FOUND)
  add_executable(hello_tbb hello_tbb.cc)
  target_link_libraries(hello_tbb TBB::tbb)
endif (TBB_FOUND)

# ===============
# Message passing
# ===============

add_executable(hello_sendrecv hello_sendrecv.cc)
target_link_libraries(hello_sendrecv Threads::Threads)

if (MPI_FOUND)
  add_executable(hello_mpi hello_mpi.cc)
  target_link_libraries(hello_mpi PRIVATE MPI::MPI_CXX)
endif (MPI_FOUND)

# ==============
# Jacobi stencil
# ==============

add_executable(jacobi_seq jacobi_seq.cc)

if (TBB_FOUND)
  add_executable(jacobi_tbb jacobi_tbb.cc)
  target_link_libraries(jacobi_tbb TBB::tbb)
endif (TBB_FOUND)

# ====
# SYCL
# ====

if (HASC_HAVE_SYCL)
  add_executable(hello_sycl hello_sycl.cc)
endif (HASC_HAVE_SYCL)
