all: spmma

spmma: spmma.cu
	nvcc spmma.cu -o spmma -lcusparse -lcusparseLt

test:
	./spmma

profile:
	nsys profile --stats=true ./spmma

clean:
	rm ./spmma *report

.PHONY: profile clean test