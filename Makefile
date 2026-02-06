all:
	cd hello && $(MAKE)
	cd matmul && $(MAKE)
	cd nbody && $(MAKE)

clean:
	cd hello && $(MAKE) clean
	cd matmul && $(MAKE) clean
	cd nbody && $(MAKE) clean

.PHONY : all clean
