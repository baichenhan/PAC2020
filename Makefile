CC	= mpiicpc
CFLAGS	= -std=c++11 -Ofast -xhost  -fp-model fast=2 -fopenmp -qopt-zmm-usage=high -ipo

all: logVS

logVS: main.o
	$(CC) -o $@ $^ $(CFLAGS)

main.o: main_optimised.cpp
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean

clean: 
	rm -f *.o
	rm -f logVS
