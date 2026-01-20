# include definitions from root directory
include ../make.def

all: hello_openmp hello_mpi hello_sendrecv hello_tbb hello_sycl

hello_openmp: hello_openmp.cc Makefile ../make.def
	$(CCOMP) $(CCFLAGSBASE) $(OMPFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

hello_mpi: hello_mpi.cc Makefile ../make.def
	$(CCMPI) $(CCFLAGSBASE) $(MPIFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSMPI)

hello_sendrecv: hello_sendrecv.cc Makefile ../cppthreads/MessageSystem.hh ../make.def
	$(CC) $(CCFLAGSBASE) -I../cppthreads/ -o $@ $< $(LFLAGS)

hello_tbb: hello_tbb.cc Makefile ../make.def
	$(CCTBB) $(CCFLAGSBASE) -o $@ $< $(LFLAGS) $(LFLAGSTBB)

hello_sycl: hello_sycl.cc Makefile ../make.def
	$(DPCPP) $(CCFLAGSBASE) $(CCFLAGSDPCPP) -o $@ $< $(LFLAGS) $(LFLAGSDPCPP)

clean:
	rm -f *.o \
	hello_openmp \
	hello_mpi \
	hello_tbb \
	hello_sycl \
	hello_sendrecv
