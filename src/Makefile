CC=nvcc
CFLAGS=
LDLIBS=
SOURCES=main.cpp gpu_impl.cu
cudaMC_pi: $(SOURCES)
	$(CC) $(LDLIBS) $(SOURCES) -o $@
	mv cudaMC_pi ../bin/
