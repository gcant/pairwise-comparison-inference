CPP=g++
CPPFLAGS=-std=c++20 -march=native -I ./inc -O3 -lfftw3 -lfftw3l -lquadmath -lm -fopenmp
SONAME=-soname

ifeq ($(shell uname -s),Darwin)
	SONAME=-install_name
	CPP=g++-15
endif

cheby-mp:
	$(CPP) $(CPPFLAGS) -c -fPIC src/cheby-mp.cpp -o out/chebyMP.o
	$(CPP) $(CPPFLAGS) -shared -Wl,$(SONAME),chebyMP.so -o out/chebyMP.so  out/chebyMP.o

