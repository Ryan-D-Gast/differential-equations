# Makefile for Fortran CR3BP Comparison Benchmarks

# Define paths to shared libraries
ROOT_DIR := ../../..
LIB_DIR := $(ROOT_DIR)/lib/fortran
DOP853_DIR := $(LIB_DIR)/dop853

jw_cr3bp: target
	gfortran -O3 -cpp -DREAL64 -J./target \
 		$(DOP853_DIR)/jw_dop853.f90 src/jw_cr3bp.f90 \
		-o target/jw_cr3bp

hw_cr3bp: target
	gfortran -O3 -w -o target/hw_cr3bp \
		src/hw_cr3bp.f $(DOP853_DIR)/hw_dop853.f

all: jw_cr3bp hw_cr3bp

target:
	@if not exist target mkdir target

run-jw:
	./target/jw_cr3bp

run-hw:
	./target/hw_cr3bp

clean:
	rm -rf target