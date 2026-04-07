# Beware!

I am constantly restructuring the code for the lecture in summer semester 2026. The best is to update the code on a regular basis. In order to keep your changes use the following commands:
```
git stash
git pull
git stash pop
```

# hasc-code

This repository collects code for the hardware aware scientific computing lecture. The code in this repository will change as the lecture proceeds so expect some changes and merge conflicts if you change anything in here.

# Software requirements

The hasc-code examples are tested on Linux (Ubuntu 24.04.3 (Noble Numbat) as well as on Apple MacBook Pro (M2 Max, MacOS Ventura, MacPorts) using the GNU C++ Compiler. Windows is not supported.

The lecture covers several different programming models. Some of these are handled by most C++ compilers, others require additional software to be installed on your system. Some of this software can be installed via usual packet managers, other has to be downloaded and installed manually. The following subsection should give the rquired information.

## Simd vectorization for Intel/AMD processors 

Explicit SIMD vectorization for AVX2 and AVX512 is done using the [vector class library](https://github.com/vectorclass/version2). Use the `--recursive` option when cloning the repository.

## Simd vectorization for Arm (neon) processors 

Explicit SIMD vectorization for Arm processors, more specifically Apple Silicon with NEON support is done using neon intrinsics [(see here for documentation)](https://github.com/thenifty/neon-guide). This should be available with gcc on such systems without installation.

## Portable Simd vectorization using std::simd 

This requires a C++-compiler supporting the 2026 standard of C++. As of this writing this is supported by GCC 14 when passing the option `-std=c++26` in the namespace `std::experimental`.

## Open MP 

Is supported by GCC and Clang compilers using appropriate options (see file `make.def`)

## Intel Thread Building Blocks (TBB)

This is an open-source library released by Intel. Hasc-code uses the newer version named `oneTBB`. It should be available with your package manager. With macports the port is called `onetbb`.

On LINUX you should install Intel's [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline), which gives you oneTBB as well as the Sycl compiler (see below).

## Sycl

[SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) is C++ extension based on an open standard that allows one to write portable code for CPUs and GPUs (at least that is the idea). [Several compilers]((https://www.intel.com/content/www/us/en/developer/articles/technical/quick-guide-to-sycl-implementations.html)) are available. Hasc-code uses Intel's [oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline) for its SYCL examples. This is restricted to Intel/AMD hardware.

After you have installed the software provided by Intel you have to execute a shell script in the shell where you want to compile in order to be able to use the Sycl compiler.

## Message Passing Interface (MPI)

Should be available with most package managers. Popular implementations are [MPICH](https://www.mpich.org/) and [OpenMPI](https://www.open-mpi.org/) (not to be confused with OpenMP).


# Hasc-code Installation

Download the code from the git repository using

```
git clone --recursive https://parcomp-git.iwr.uni-heidelberg.de/Teaching/hasc-code.git
```
The `--recursive` option is important to include the [vector class library](https://github.com/vectorclass/version2) as submodule.


Have a look at the file `make.def` in the top level directory and adjust it to your system.

- Set your architecture
- Select the available programming models
- Choose compilers
- Adjust the compiler flags
- Adjust the linker flags

If everything is set up correctly you should be able to build the examples by just typing

```
make
```

in the top-level directory.
