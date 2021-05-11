# define your compiler
CC = g++-mp-9
# CC = g++-8
# CC = g++-10

# compilation flags without GMP stuff
# no vectorization
//CCFLAGS = -O0
//CCFLAGS = -O3 -fno-tree-vectorize -fno-trapping-math -fno-math-errno -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias
# AVX2 with vector class library
CCFLAGS = -std=c++17 -O3 -mavx2 -mfma -fno-trapping-math -fno-math-errno -fabi-version=0 -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias

# linker flags without GMP stuff
LFLAGS = -lm
# LFLAGS = -lm -pthread # In case you need to link against pthread

all: scalar_product_v1\
     scalar_product_v0\
     scalar_product_v2\
     scalar_product_v3\
     scalar_product_v4\
     scalar_product_v5\
     matmul_seq_v1\
     matmul_seq_v2\
     pointer_chasing\
     transpose_v1\
     matvec_v1 \
     matvec_v2 \
     peterson\
     philosophers\
     barrier\
     packaged_task\
     producer_consumer\
     nbody_vanilla\
     nbody_optimized

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
matmul_seq_v1: matmul_seq_v1.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
matmul_seq_v2: matmul_seq_v2.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
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
nbody_optimized: nbody_optimized.cc Makefile nbody_generate.hh nbody_io.hh
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
peterson: peterson.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
philosophers: philosophers.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
barrier: barrier.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
packaged_task: packaged_task.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)
producer_consumer: producer_consumer.cc Makefile
	$(CC) $(CCFLAGS) -o $@ $< $(LFLAGS)

clean:
	rm -f *.o \
	scalar_product_v1 \
        scalar_product_v0 \
        scalar_product_v2 \
        scalar_product_v3 \
        scalar_product_v4 \
        scalar_product_v5 \
        matmul_seq_v1 \
        matmul_seq_v2 \
        pointer_chasing \
        transpose_v1\
	matvec_v1\
	matvec_v2\
	peterson\
	philosophers\
	barrier\
	packaged_task\
	producer_consumer\
	nbody_vanilla\
	nbody_ooptimized
