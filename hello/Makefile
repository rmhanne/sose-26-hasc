# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = hello_openmp hello_sendrecv 
MPI_TARGETS = hello_mpi 
TBB_TARGETS = hello_tbb
SYCL_TARGETS = hello_sycl

TARGETS = $(GENERIC_TARGETS)
ifdef HAVE_MPI
	TARGETS += $(MPI_TARGETS)
endif
ifdef HAVE_TBB
	TARGETS += $(TBB_TARGETS)
endif
ifdef HAVE_SYCL
	TARGETS += $(SYCL_TARGETS)
endif

all: $(TARGETS)

hello_openmp: hello_openmp.cc Makefile ../make.def
	$(CCOMP) $(CCFLAGSBASE) $(OMPFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

hello_mpi: hello_mpi.cc Makefile ../make.def
	$(CCMPI) $(CCFLAGSBASE) $(MPIFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSMPI)

hello_sendrecv: hello_sendrecv.cc Makefile ../cppthreads/MessageSystem.hh ../make.def
	$(CC) $(CCFLAGSBASE) -I../cppthreads/ -o $@ $< $(LFLAGS)

hello_tbb: hello_tbb.cc Makefile ../make.def
	$(CCTBB) $(CCFLAGSBASE) $(TBBFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSTBB)

hello_sycl: hello_sycl.cc Makefile ../make.def
	$(DPCPP) $(CCFLAGSBASE) $(CCFLAGSDPCPP) -o $@ $< $(LFLAGS) $(LFLAGSDPCPP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)
