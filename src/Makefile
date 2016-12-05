EXEC=cudaMC_pi
CC=nvcc
CFLAGS=
LDLIBS=
HEADERS=
SOURCES=main.cpp gpu_impl.cu
$(EXEC): $(SOURCES) thrustMC
	$(CC) $(LDLIBS) $(SOURCES) -o $@
	mv cudaMC_pi ../bin/
thrustMC: monte_carlo.cu
	$(CC) $(LDLIBS) $^ -o $@
	mv thrustMC ../bin/
.PHONY: clean
clean:
	rm ../bin/$(EXEC)
	rm ../bin/thrustMC
