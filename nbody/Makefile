# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = nbody_vanilla
ARM_TARGETS = nbody_armneon_SoA nbody_omp_armneon_SoA
MPI_TARGETS =
TBB_TARGETS =
SYCL_TARGETS =

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
ifeq ($(ARCHITECTURE),arm)
	TARGETS += $(ARM_TARGETS)
endif

# include path for all targets
ROOTPATH = ../

all: $(TARGETS)

nbody_vanilla: nbody_vanilla.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

nbody_armneon_SoA: nbody_armneon_SoA.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

nbody_omp_armneon_SoA: nbody_omp_armneon_SoA.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
