# define your compiler
#CC = g++-mp-9
# CC = g++-8
CC = g++-mp-10

# compilation flags without GMP stuff
# no vectorization
#CCFLAGS = -O0
#CCFLAGS = -O3 -fno-tree-vectorize -fno-trapping-math -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias
# AVX2 with vector class library
CCFLAGS_NOVEC = -std=c++20 -O3 -ffast-math -fargument-noalias
CCFLAGS = -std=c++20 -O3 -mavx2 -mfma -fno-trapping-math -fabi-version=0 -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias
CCFLAGS_OMP = -fopenmp -std=c++20 -O3 -mavx2 -mfma -fno-trapping-math -fabi-version=0 -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias

# linker flags without GMP stuff
LFLAGS = -lm -lpthread
LFLAGS_OMP = -lm -lpthread


all: scalar_product_v1\
     scalar_product_v0\
     scalar_product_v2\
     scalar_product_v3\
     scalar_product_v4\
     scalar_product_v5\
     scalar_product_ms\
     matmul_seq_v1\
     matmul_seq_v2\
     matmul_omp\
     pointer_chasing\
     transpose_v1\
     matvec_v1 \
     matvec_v2 \
     peterson\
     philosophers\
     barrier\
     power_method\
     packaged_task\
     producer_consumer\
     nbody_vanilla\
     nbody_vectorized\
     nbody_omp\
     nbody_vectorized_threaded\
     jacobi_seq\
     hello_openmp\
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
scalar_product_ms: scalar_product_ms.cc Makefile MessageSystem.hh Barrier.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_seq_v1: matmul_seq_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_seq_v2: matmul_seq_v2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_omp: matmul_omp.cc Makefile
	$(CC) $(CCFLAGS_OMP) -o $@ $< $(LFLAGS_OMP)
pointer_chasing: pointer_chasing.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
transpose_v1: transpose_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matvec_v1: matvec_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matvec_v2: matvec_v2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_vanilla: nbody_vanilla.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_vectorized: nbody_vectorized.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_vectorized_threaded: nbody_vectorized_threaded.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
nbody_omp: nbody_omp.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS_OMP) -o $@ $< $(LFLAGS_OMP)
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
hello_openmp: hello_openmp.cc Makefile
	$(CC) $(CCFLAGS_OMP) -o $@ $< $(LFLAGS_OMP)
hello_sendrecv: hello_sendrecv.cc Makefile MessageSystem.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)

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
        pointer_chasing \
        transpose_v1 \
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
	nbody_omp \
	nbody_vectorized_threaded \
	jacobi_seq \
	hello_openmp \
	hello_sendrecv
