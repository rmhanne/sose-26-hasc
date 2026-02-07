# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = matmul_seq	matmul_omp_stdsimd
ARM_TARGETS = matmul_omp_armneon
INTEL_TARGETS = matmul_omp_avx512

ifeq ($(ARCHITECTURE),generic)
	TARGETS = $(GENERIC_TARGETS)
endif
ifeq ($(ARCHITECTURE),intel)
	TARGETS = $(GENERIC_TARGETS) $(INTEL_TARGETS)
endif
ifeq ($(ARCHITECTURE),arm)
	TARGETS = $(GENERIC_TARGETS) $(ARM_TARGETS)
endif

# include path for all targets
ROOTPATH = ../

# the default target
all: $(TARGETS)

matmul_seq: matmul_seq.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SCALARFLAGS) -o $@ $< $(LFLAGS)

matmul_omp_stdsimd: matmul_omp_stdsimd.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(STDSIMDFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_omp_avx512: matmul_omp_avx512.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(AVX512FLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_omp_armneon: matmul_omp_armneon.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
