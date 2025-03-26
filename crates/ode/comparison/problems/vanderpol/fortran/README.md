# Fortran DOP853 Van der Pol Oscillator Benchmark

This directory contains a Fortran implementation of the DOP853 solver applied to the various problems. The DOP853 Solver is the flagship solver of `rgode` and also the most important and highest accuracy solver likely to be the main solver used by users. Thus systems will be compared against the Fortran implementation of the DOP853 solver.

## Fortran Implementations

First there is a modern Fortran implementation of the DOP853 solver by Jacob Williams. Which can be found at [https://github.com/jacobwilliams/dop853](github.com/jacobwilliams/dop853). 

The second implementation is by the original authors of the DOP853 solver, Hairer and Wanner. This implementation can by found at [https://www.unige.ch/~hairer/software.html](https://www.unige.ch/~hairer/software.html).

## Building

Using the provided Makefile and gfortran:

- `make jw_vanderpol` will build the Jacob Williams implementation of the DOP853 solver applied to the Van der Pol Oscillator.

- `make hw_vanderpol` will build the Hairer and Wanner implementation of the DOP853 solver applied to the Van der Pol Oscillator.

- `make all` will build both implementations.

The code can be run by calling the executables in the `benches/fortran_comparison/target/` directory.