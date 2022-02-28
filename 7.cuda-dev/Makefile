all: spmma spmma_static

spmma: spmma.cpp
	nvcc spmma.cpp -o spmma -lcusparseLt -lcusparse

# -Xlinker=--whole-archive: 将后面库中所有符号都链接进来
# -Xlinker=--no-whole-archive: 避免后面库的所有符号被链接进来
spmma_static: spmma.cpp
	nvcc spmma.cpp -o spmma_static -Xlinker=--whole-archive -Xlinker=/lib64/libcusparseLt_static.a -Xlinker=--no-whole-archive -lcusparse

profile:
	nsys profile --stats=true ./spmma

clean:
	rm -f spmma spmma_static spmma.o a* spmma_d* report* *.bin

test:
	@echo "\n==== SpMMA Test ====\n"
	./spmma

# 在Makefile中，.PHONY后面的target表示的也是一个伪造的target, 而不是真实存在的文件target，注意Makefile的target默认是文件
.PHONY: clean all test profile
