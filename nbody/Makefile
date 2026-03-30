# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = nbody_vanilla
OPENMP_TARGETS = nbody_omp_SoA nbody_omp_stdsimd_SoA
ARM_TARGETS = nbody_armneon_SoA nbody_omp_armneon_SoA
INTEL_TARGETS = nbody_omp_intel_SoA
MPI_TARGETS =
TBB_TARGETS =
SYCL_TARGETS =

TARGETS = $(GENERIC_TARGETS)
ifdef HAVE_OPENMP
	TARGETS += $(OPENMP_TARGETS)
endif
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
ifeq ($(ARCHITECTURE),intel)
	TARGETS += $(INTEL_TARGETS)
endif

# include path for all targets
ROOTPATH = ../

all: $(TARGETS)

nbody_vanilla: nbody_vanilla.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

nbody_omp_SoA: nbody_omp_SoA.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(VECTORIZATIONFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

nbody_armneon_SoA: nbody_armneon_SoA.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

nbody_omp_armneon_SoA: nbody_omp_armneon_SoA.cc Makefile ../make.def nbody_generate.hh nbody_io.hh
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

nbody_omp_intel_SoA: nbody_omp_intel_SoA.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(AVX512FLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

nbody_omp_stdsimd_SoA: nbody_omp_stdsimd_SoA.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(STDSIMDFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
