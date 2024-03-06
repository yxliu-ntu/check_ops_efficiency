CXX = gcc
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = gcc
endif
dtype ?= float

#To init MKLROOT just Do  "dot white dot/init.sh". That is, ". ./init.sh" 
#MKLROOT = /home/ybw/intel/compilers_and_libraries_2018.2.199/linux/mkl

CXXFLAGS = -Wall -O3 -march=native -fPIC
#MKLFLAGS = -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
CXXFLAGS += -fopenmp
#CXXFLAGS += $(MKLFLAGS)

all: cfuns.so

cfuns.so: cfuns.c
	$(CXX) $(CXXFLAGS)  -shared -Ddtype=$(dtype) -o $@ $<

clean:
	rm -f *.so
