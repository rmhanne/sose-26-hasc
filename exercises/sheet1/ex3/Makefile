# include definitions from root directory
include ../../../make.def

# define targets
GENERIC_TARGETS = compiler_flags compiler_flags_00 compiler_flags_01 compiler_flags_02 compiler_flags_03 compiler_flags_native compiler_flags_funroll compiler_flags_ffast compiler_flags_ffast_native compiler_flags_funroll_native compiler_flags_ffast_funroll
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

compiler_flags: compiler-flags.cc Makefile ../../../make.def
	$(CC) -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_00: compiler-flags.cc Makefile ../../../make.def
	$(CC) -I$(ROOTPATH) -O0 -o $@ $< $(LFLAGS)

compiler_flags_01: compiler-flags.cc Makefile ../../../make.def
	$(CC) -I$(ROOTPATH) -O1 -o $@ $< $(LFLAGS)

compiler_flags_02: compiler-flags.cc Makefile ../../../make.def
	$(CC) -O2 -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_03: compiler-flags.cc Makefile ../../../make.def
	$(CC) -O3 -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_native: compiler-flags.cc Makefile ../../../make.def
	$(CC) -march=native -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_funroll: compiler-flags.cc Makefile ../../../make.def
	$(CC) -funroll-loops -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_ffast: compiler-flags.cc Makefile ../../../make.def
	$(CC) -ffast-math -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_ffast_native: compiler-flags.cc Makefile ../../../make.def
	$(CC) -march=native -ffast-math -I$(ROOTPATH) -o $@ $< $(LFLAGS)


compiler_flags_funroll_native: compiler-flags.cc Makefile ../../../make.def
	$(CC) -funroll-loops -march=native -I$(ROOTPATH) -o $@ $< $(LFLAGS)

compiler_flags_ffast_funroll: compiler-flags.cc Makefile ../../../make.def
	$(CC) -funroll-loops -ffast-math -I$(ROOTPATH) -o $@ $< $(LFLAGS)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
