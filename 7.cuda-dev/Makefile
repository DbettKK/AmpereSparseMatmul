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
# 原LIBS多余 只需要一项即可
# LIBS         := -lcudart -lcusparse -ldl ${NVRTC_SHARED}
LIBS         := -lcusparse

all: spmma_example spmma_example_static

spmma_example: spmma_example.cpp
	nvcc --std=c++14  ${INCS}  spmma_example.cpp -o spmma_example  \
         -L${CUSPARSELT_PATH}/lib64  -lcusparseLt ${LIBS}

# -Xlinker=--whole-archive: 将后面库中所有符号都链接进来
# -Xlinker=--no-whole-archive: 避免后面库的所有符号被链接进来
spmma_example_static: spmma_example.cpp
	nvcc --std=c++14 ${INCS} spmma_example.cpp -o spmma_example_static \
         -Xlinker=--whole-archive                                      \
         -Xlinker=${CUSPARSELT_PATH}/lib64/libcusparseLt_static.a      \
         -Xlinker=--no-whole-archive                                   \
         ${LIBS}

clean:
	rm -f spmma_example spmma_example_static spmma_example.o a* spmma_example_d*

test:
	@echo "\n==== cusparseLt SpMMA Test ====\n"
	./spmma_example

.PHONY: clean all test

# 命令的呈现
# nvcc --std=c++14  -I/usr/local/cuda-11.4/include spmma_example.cpp -o spmma_example -L/lib64  -lcusparseLt -lcudart -lcusparse -ldl /usr/local/cuda-11.4/targets/x86_64-linux/lib/libnvrtc.so
# 简化版
# nvcc spmma_example.cpp -lcusparseLt -lcusparse
# static版本
# nvcc  spmma_example.cpp -o spmma_example_static -Xlinker=--whole-archive -Xlinker=/lib64/libcusparseLt_static.a -Xlinker=--no-whole-archive -lcudart -lcusparse                                  \
# 生成ptx
# cuobjdump -ptx spmma_example_static > ~/t2.txt

