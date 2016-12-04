CC=nvcc
CFLAGS=
LDLIBS=
SOURCES=main.cpp gpu_impl.cu
OBJECTS=$(SOURCES:.cpp=.o)
cudaMC_pi: $(OBJECTS)
	$(CC) $(LDLIBS) $(OBJECTS) -o $@
all:
	$(SOURCES) 	 
