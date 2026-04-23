all:
	cd hello && $(MAKE)
	cd scalar_product && $(MAKE)
	cd matmul && $(MAKE)
	cd nbody && $(MAKE)
	cd benchmarking && $(MAKE)
	cd stream && $(MAKE)
	cd exercises && $(MAKE)

exercises:
	cd exercises && $(MAKE)

clean:
	cd hello && $(MAKE) clean
	cd scalar_product && $(MAKE) clean
	cd matmul && $(MAKE) clean
	cd nbody && $(MAKE) clean
	cd benchmarking && $(MAKE) clean
	cd stream && $(MAKE) clean

.PHONY : all clean
