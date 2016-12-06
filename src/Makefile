EXEC=cudaMC_pi
CC=nvcc
CFLAGS=
LDLIBS=
HEADERS=
SOURCES=main.cpp gpu_impl.cu monte_carlo.cu gpu_impl_noVal.cu
$(EXEC): $(SOURCES) 
	$(CC) $(LDLIBS) $(SOURCES) -o $@
	mv cudaMC_pi ../bin/
.PHONY: clean
clean:
	rm ../bin/$(EXEC)
	rm ../bin/thrustMC
