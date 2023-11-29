# define your compiler
#CC = g++-mp-9
# CC = g++-8
CC = g++
CCMPI = mpicxx
CCTBB = g++
DPCPP = dpcpp

# compilation flags without GMP stuff
# no vectorization
#CCFLAGS = -O0
#CCFLAGS = -O3 -fno-tree-vectorize -fno-trapping-math -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias
# AVX2 with vector class library
CCFLAGSBASE2 = -std=c++17 -O3
CCFLAGSBASE = -std=c++17 -O3 -fno-trapping-math -fabi-version=0 -funroll-loops -ffast-math -fargument-noalias
ARMFLAGS = -ftree-vectorize -march=armv8-a+dotprod -fopt-info-vec
SSE2FLAGS = -ftree-vectorize -msse2 -fopt-info-vec
AVX2FLAGS = -ftree-vectorize -mavx2 -mfma -fopt-info-vec
AVX512FLAGS = -ftree-vectorize -mfma -mavx512f -mavx512cd -march=skylake-avx512 -flto
OMPFLAGS = -fopenmp
CCFLAGS = $(CCFLAGSBASE) $(ARMFLAGS)
CCFLAGS_AVX2 = $(CCFLAGSBASE) $(AVX2FLAGS)
CCFLAGS_AVX512 = $(CCFLAGS_BASE) $(AVX512FLAGS)
CCFLAGS_DPCPP = -Ofast -fargument-noalias

//CCFLAGS_TBB = -std=c++17 -Ofast -xHost -fargument-noalias

# linker flags
LFLAGS = -lm -lpthread
LFLAGS_OMP = -lm -lpthread
LFLAGS_MPI = -lm -lpthread
LFLAGS_TBB = -lm -ltbb
LFLAGS_DPCPP =


all: scalar_product_v1\
     scalar_product_v0\
     scalar_product_v2\
     scalar_product_v3\
     scalar_product_v4\
     scalar_product_v5\
     scalar_product_ms\
     scalar_product_faster\
     matmul_seq_v1\
     matmul_M2\
     matmul_seq_v2\
     matmul_omp\
     matmul_omp_avx512\
     matmul_omp_milan\
     pointer_chasing\
     transpose\
		 transpose_avx\
     transpose_neon\
     matvec_v1 \
     matvec_v2 \
     peterson\
     philosophers\
     barrier\
     power_method\
     packaged_task\
     producer_consumer\
     nbody_vanilla\
     nbody_avx\
     nbody_neon\
     nbody_neon_v2\
     nbody_intel_SoA\
     nbody_mpi\
     nbody_mpi_nonblocking\
     nbody_omp\
     nbody_mpi_omp\
     nbody_vectorized_threaded\
     nbody_tbb_v2\
     nbody_sycl\
     jacobi_seq\
     jacobi_tbb\
     lu\
     hello_openmp\
     hello_mpi\
     hello_tbb\
     hello_sycl\
     hello_sendrecv

scalar_product_v0: scalar_product_v0.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_v1: scalar_product_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_v2: scalar_product_v2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_v3: scalar_product_v3.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_v4: scalar_product_v4.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_v5: scalar_product_v5.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_neon: scalar_product_neon.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
scalar_product_ms: scalar_product_ms.cc Makefile MessageSystem.hh Barrier.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_seq_v1: matmul_seq_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_M2: matmul_M2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_seq_v2: matmul_seq_v2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_omp: matmul_omp.cc Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS_OMP)
matmul_omp_avx512: matmul_omp_avx512.cc Makefile
	$(CC) $(CCFLAGS_AVX512) $(OMPFLAGS) -o $@ $< $(LFLAGS_OMP)
matmul_omp_milan: matmul_omp_milan.cc Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS_OMP)
lu: lu.cc Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS)
pointer_chasing: pointer_chasing.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
transpose: transpose.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
transpose_avx: transpose_avx.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
transpose_neon: transpose_neon.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matvec_v1: matvec_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matvec_v2: matvec_v2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_vanilla: nbody_vanilla.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGSBASE) -o $@ $< $(LFLAGS)
nbody_vectorized: nbody_vectorized.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_intel_SoA: nbody_intel_SoA.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_neon: nbody_neon.cc Makefile
	$(CC) $(CCFLAGSBASE) -o $@ $< $(LFLAGS)
nbody_neon_v2: nbody_neon_v2.cc Makefile
	$(CC) $(CCFLAGSBASE) $(ARMFLAGS) -o $@ $< $(LFLAGS)
nbody_avx: nbody_avx.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS)
nbody_vectorized_threaded: nbody_vectorized_threaded.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_tbb_v2: nbody_tbb_v2.cc Makefile nbody_generate.hh nbody_io.hh
	$(CCTBB) $(CCFLAGS) -o $@ $< $(LFLAGS_TBB)
nbody_omp: nbody_omp.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS_OMP)
nbody_mpi: nbody_mpi.cc Makefile nbody_generate.hh nbody_io.hh
	$(CCMPI) $(CCFLAGS) -o $@ $< $(LFLAGS_MPI)
nbody_mpi_nonblocking: nbody_mpi_nonblocking.cc Makefile nbody_generate.hh nbody_io.hh
	$(CCMPI) $(CCFLAGS) -o $@ $< $(LFLAGS_MPI)
nbody_mpi_omp: nbody_mpi_omp.cc Makefile nbody_generate.hh nbody_io.hh
	$(CCMPI) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS_MPI) $(LFLAGS_OMP)
peterson: peterson.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
philosophers: philosophers.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
barrier: barrier.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
power_method: power_method.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
packaged_task: packaged_task.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
producer_consumer: producer_consumer.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
jacobi_seq: jacobi_seq.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
jacobi_tbb: jacobi_tbb.cc Makefile
	$(CCTBB) $(CCFLAGS) -o $@ $< $(LFLAGS_TBB)
hello_openmp: hello_openmp.cc Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $< $(LFLAGS_OMP)
hello_sendrecv: hello_sendrecv.cc Makefile MessageSystem.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
hello_mpi: hello_mpi.cc Makefile
	$(CCMPI) $(CCFLAGS) -o $@ $< $(LFLAGS_MPI)
hello_tbb: hello_tbb.cc Makefile
	$(CCTBB) $(CCFLAGS) -o $@ $< $(LFLAGS_TBB)
hello_sycl: hello_sycl.cc Makefile
	$(DPCPP) $(CCFLAGS_DPCPP) -o $@ $< $(LFLAGS_DPCPP)
nbody_sycl: nbody_sycl.cc Makefile nbody_generate.hh nbody_io.hh
	$(DPCPP) $(CCFLAGS_DPCPP) -o $@ $< $(LFLAGS_DPCPP)

clean:
	rm -f *.o \
	scalar_product_v1 \
        scalar_product_v0 \
        scalar_product_v2 \
        scalar_product_v3 \
        scalar_product_v4 \
        scalar_product_v5 \
        scalar_product_ms \
        matmul_seq_v1 \
        matmul_seq_v2 \
        matmul_omp \
        matmul_omp_avx512 \
        pointer_chasing \
        transpose \
				transpose_avx \
				transpose_neon \
	matvec_v1 \
	matvec_v2 \
	peterson \
	philosophers \
	barrier \
	power_method \
	packaged_task \
	producer_consumer \
	nbody_vanilla \
	nbody_vectorized \
	nbody_intel_SoA \
	nbody_omp \
	nbody_mpi \
	nbody_vectorized_threaded \
	nbody_tbb_v2 \
	jacobi_seq \
	jacobi_tbb \
	lu \
	hello_openmp \
	hello_mpi \
	hello_tbb \
	hello_sycl \
	hello_sendrecv
