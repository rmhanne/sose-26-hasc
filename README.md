# hasc-code

This repository collects code for the hardware aware scientific computing lecture. The code in this repository will change as the lecture proceeds so expect some changes and merge conflicts if you change anything in here.

You can either directly use the Makefile provided in this repository or use CMake for building the code.

## Makefile

Have a look at the `Makefile` and adjust it to your system:

- Choose a compiler
- Adjust the compiler flags
- Adjust the linker flags

If everything is set up correctly you should be able to build the examples by typing

```
make
```

## CMake

As an alternative you can use CMake to create an out of source build. If you want to change the compiler flags you can do so in the file `CMakeLists.txt`. The steps to build the executables are as follows:

- Create a build directory and go there, eg `mkdir build; cd build`
- Call cmake and provide the path to `hasc-code`, in our example `cmake ..`
- Build the programs by typing `make`

## Troubleshooting

In case of problems you can try to build the executables with

```
make VERBOSE=1
```

This way you will see all commands that get executed. This way you can check if the compiler and the compile and link flags look reasonable.
