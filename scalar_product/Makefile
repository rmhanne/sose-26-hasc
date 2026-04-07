# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = scalar_product_v0 scalar_product_v1 scalar_product_v2 scalar_product_v3 scalar_product_v4 scalar_product_v5 scalar_product_ms
OPENMP_TARGETS = 
ARM_TARGETS = scalar_product_neon
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
ROOTPATH = ../

# the default target
all: $(TARGETS)

scalar_product_v0: scalar_product_v0.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_v1: scalar_product_v1.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_v2: scalar_product_v2.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_v3: scalar_product_v3.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_v4: scalar_product_v4.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_v5: scalar_product_v5.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_ms: scalar_product_ms.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -I$(ROOTPATH)cppthreads $(CCFLAGSBASE) $(SIMDFLAGS) -o $@ $< $(LFLAGS)

scalar_product_neon: scalar_product_neon.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
