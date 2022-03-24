#include<iostream>
#include<cuda_fp16.h>

using namespace std;

template <typename Dtype>
void test_right(const Dtype *item1, const Dtype *item2, int total) {
    int cnt = 0;
    printf("total: %d\n", total);
    for (int i = 0; i < total; i++) {
        if (typeid(item1) == typeid(__half *)) {
            if (item1[i] != item2[i]) {
                cnt++;
                printf("%d : %d\n", __half2int_rz(item[i]), __half2int_rz(item[i]));
            }
        } else {
            if (item1[i] != item2[i]) {
                cnt++;
                printf("%f : %f\n", item[i], item[i]);
            }
        }

    }
    printf("diff: %d\n", cnt);
}