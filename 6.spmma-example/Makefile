CUDA_TOOLKIT := $(abspath $(shell dirname $$(command -v nvcc))/..)
ifeq ($(shell uname -m), aarch64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "sbsa-linux"
endif
endif
ifeq ($(shell uname -m), x86_64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "x86_64-linux"
endif
endif
NVRTC_SHARED := ${CUDA_TOOLKIT}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so
INCS         := -I$(CUDA_TOOLKIT)/include -I${CUSPARSELT_PATH}/include
LIBS         := -lcudart -lcusparse -ldl ${NVRTC_SHARED}

all: spmma_example spmma_example_static

spmma_example: spmma_example.cpp
	nvcc --std=c++14  ${INCS}  spmma_example.cpp -o spmma_example  \
         -L${CUSPARSELT_PATH}/lib64  -lcusparseLt ${LIBS}

spmma_example_static: spmma_example.cpp
	nvcc --std=c++14 ${INCS} spmma_example.cpp -o spmma_example_static \
         -Xlinker=--whole-archive                                      \
         -Xlinker=${CUSPARSELT_PATH}/lib64/libcusparseLt_static.a      \
         -Xlinker=--no-whole-archive                                   \
         ${LIBS}

clean:
	rm -f spmma_example spmma_example_static

test:
	@echo "\n==== cusparseLt SpMMA Test ====\n"
	./spmma_example

.PHONY: clean all test