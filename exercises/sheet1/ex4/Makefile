# include definitions from root directory
include ../../../make.def

# define targets
GENERIC_TARGETS = blas
OPENMP_TARGETS =
ARM_TARGETS =
INTEL_TARGETS =
SYCL_TARGETS =

TARGETS = $(GENERIC_TARGETS)
ifdef HAVE_OPENMP
	TARGETS += $(OPENMP_TARGETS)
endif
ifeq ($(ARCHITECTURE),intel)
	TARGETS += $(INTEL_TARGETS)
endif
ifeq ($(ARCHITECTURE),arm)
	TARGETS += $(ARM_TARGETS)
endif
ifdef HAVE_SYCL
	TARGETS += $(SYCL_TARGETS)
endif

# include path for all targets
ROOTPATH = ../../

# the default target
all: $(TARGETS)

blas: blas.cc Makefile ../../../make.def
	$(CC) -I$(ROOTPATH) $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
