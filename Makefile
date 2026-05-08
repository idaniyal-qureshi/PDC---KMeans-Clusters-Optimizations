CXX = g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -fopenmp

# OpenCL configuration (can be overridden)
OCL_LDFLAGS = -lOpenCL

TARGETS = main_seq main_par main_opt main_bench main_opencl

all: $(TARGETS)

main_seq: main_sequential.cpp Point.h Cluster.h
	$(CXX) $(CXXFLAGS) -o $@ main_sequential.cpp

main_par: main_parallel.cpp Point.h Cluster.h
	$(CXX) $(CXXFLAGS) -o $@ main_parallel.cpp

main_opt: main_optimized.cpp KMeansOptimized.h
	$(CXX) $(CXXFLAGS) -o $@ main_optimized.cpp

main_bench: main_bench.cpp KMeansOptimized.h Point.h Cluster.h
	$(CXX) $(CXXFLAGS) -o $@ main_bench.cpp

main_opencl: main_opencl.cpp
	-$(CXX) $(CXXFLAGS) -o $@ main_opencl.cpp $(OCL_LDFLAGS) || echo "Warning: OpenCL build failed (OpenCL libraries might be missing)"

clean:
	rm -f $(TARGETS) data.txt data_opt.txt

.PHONY: all clean seq par opt bench opencl

seq: main_seq
par: main_par
opt: main_opt
bench: main_bench
opencl: main_opencl
