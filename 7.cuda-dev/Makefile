all: spmma spmma_oo spmma_static cudnn cublas

spmma: spmma.cpp
	nvcc spmma.cpp -o spmma -lcusparseLt -lcusparse

spmma_oo: spmma_oo.cpp
	nvcc spmma_oo.cpp -o spmma_oo -lcusparseLt -lcusparse

cudnn: cudnn_conv.cpp
	nvcc cudnn_conv.cpp -o cudnn_conv -lcudnn

cublas: cublas_gemm.cpp
	nvcc cublas_gemm.cpp -o cublas_gemm -lcublas

data: generate_data.py
	python3 generate_data.py

# -Xlinker=--whole-archive: 将后面库中所有符号都链接进来
# -Xlinker=--no-whole-archive: 避免后面库的所有符号被链接进来
spmma_static: spmma.cpp
	nvcc spmma.cpp -o spmma_static -Xlinker=--whole-archive -Xlinker=/lib64/libcusparseLt_static.a -Xlinker=--no-whole-archive -lcusparse

profile_spmma:
	nsys profile --stats=true ./spmma

profile_spmma_oo:
	nsys profile --stats=true ./spmma_oo

profile_cudnn:
	nsys profile --stats=true ./cudnn_conv

profile_cublas:
	nsys profile --stats=true ./cublas_gemm

clean:
	rm -f spmma spmma_static report* cudnn_conv cublas_gemm

test:
	@echo "\n==== SpMMA Test ====\n"
	./spmma
	@echo "\n\n==== CUDNN Test ====\n"
	./cudnn_conv
	@echo "\n\n==== CUBLAS Test ====\n"
	./cublas_gemm

# 在Makefile中，.PHONY后面的target表示的也是一个伪造的target, 而不是真实存在的文件target，注意Makefile的target默认是文件
.PHONY: clean all test profile_spmma profile_spmma_oo profile_cudnn profile_cublas
