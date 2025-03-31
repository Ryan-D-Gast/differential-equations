# Makefile for Fortran DOP853 Comparison Benchmarks

# Define paths to shared libraries
ROOT_DIR := ../../..
LIB_DIR := $(ROOT_DIR)/lib/fortran
DOP853_DIR := $(LIB_DIR)/dop853

jw_vanderpol: target
	gfortran -O3 -march=native -cpp -DREAL64 -J./target \
		$(DOP853_DIR)/jw_dop853.f90 src/jw_vanderpol.f90 \
		-o target/jw_vanderpol

hw_vanderpol: target
	gfortran -O3 -march=native -w -o target/hw_vanderpol \
		src/hw_vanderpol.f $(DOP853_DIR)/hw_dop853.f

all: jw_vanderpol hw_vanderpol

target:
	"mkdir" -p "target"

run-jw:
	./target/jw_vanderpol

run-hw:
	./target/hw_vanderpol

clean:
	rm -rf target