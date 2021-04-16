# define your compiler
CC = g++-mp-9
# CC = g++-8
# CC = g++-10

# compilation flags without GMP stuff
#CCFLAGS = -O3 -march=native -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias
CCFLAGS = -O3 -funroll-loops -ffast-math -fopt-info-vec -fargument-noalias
#CCFLAGS = -O0

# linker flags without GMP stuff
LFLAGS = -lm
# LFLAGS = -lm -pthread # In case you need to link against pthread

all: scalar_product_v1\
     scalar_product_v0\
     scalar_product_v2\
     scalar_product_v3\
     scalar_product_v4\
     matmul_seq_v1\
     pointer_chasing\
     scalar_product_v5

scalar_product_v0: scalar_product_v0.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
scalar_product_v1: scalar_product_v1.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
scalar_product_v2: scalar_product_v2.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
scalar_product_v3: scalar_product_v3.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
scalar_product_v4: scalar_product_v4.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
scalar_product_v5: scalar_product_v5.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
matmul_seq_v1: matmul_seq_v1.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
pointer_chasing: pointer_chasing.cc
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

clean:
	rm -f *.o \
        scalar_product_v1 \
        scalar_product_v0 \
        scalar_product_v2 \
        scalar_product_v3 \
        scalar_product_v4 \
        matmul_seq_v1 \
        pointer_chasing \
        scalar_product_v5
