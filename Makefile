CC=nvcc
ifdef ALT_CC
	CC:=$(ALT_CC)
endif
DEFAULT_REAL=float
ifndef REAL
	REAL=$(DEFAULT_REAL)
endif
CFLAGS=-I$(IDIR) --std=c++11 -DREAL_TYPE=$(REAL)
ifeq ($(CC), nvcc)
	CFLAGS+= -Xcompiler -fopenmp
else
	CFLAGS+= -fopenmp
endif
IDIR=lib
LIBS=-L/usr/lib/openmpi -lmpi -lmpi_cxx
_DEPS=*.cu*
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))
MACROS=
ifeq ($(CC), nvcc)
	LANG=cu
else
	LANG=c++
endif

ODIR=obj

_OBJ = main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ -x $(LANG) $< $(CFLAGS) `libpng-config --cflags`

all: makedir dmbrot

dmbrot: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) `libpng-config --ldflags`

#openmp: CC := g++
#openmp: CFLAGS += -fopenmp
#openmp: dmbrot

#cuda: CC := nvcc
#openmp: MACROS += __CUDACC__
#cuda: dmbrot


debug: CFLAGS += -DDEBUG -g
debug: dmbrot

makedir:
	mkdir -p obj
.PHONY: clean 
clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
