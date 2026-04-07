all:
	cd hello && $(MAKE)
	cd scalar_product && $(MAKE)
	cd matmul && $(MAKE)
	cd nbody && $(MAKE)

clean:
	cd hello && $(MAKE) clean
	cd scalar_product && $(MAKE) clean
	cd matmul && $(MAKE) clean
	cd nbody && $(MAKE) clean

.PHONY : all clean
