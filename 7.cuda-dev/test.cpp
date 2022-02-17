#include<iostream>
#include<cstring>
#include<cstdio>
#include<cstdlib>

using namespace std;

typedef int __half;

__half *handle_input(__half *item, int m, int n) {
    if (m % 8 == 0 && n % 8 == 0) {
        return item;
    }
    if (m % 8 == 0) {
        int fix = 8 - n % 8;
        __half *ret = (__half *)malloc(m * (n + fix) * sizeof(__half));
        int ret_cnt = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ret[ret_cnt++] = item[i * n + j];
            }
            for (int j = 0; j < fix; j++) {
                ret[ret_cnt++] = 0;
            }
        }
        return ret;
    }
    if (n % 8 == 0) {
        int fix = 8 - m % 8;
        __half *ret = (__half *)malloc((m + fix) * n * sizeof(__half));
        memset(ret, 0, (m + fix) * n * sizeof(__half));
        memcpy(ret, item, m * n * sizeof(__half));
        return ret;
    }
    int fix_m = 8 - m % 8;
    int fix_n = 8 - n % 8;
    __half *ret = (__half *)malloc((m + fix_m) * (n + fix_n) * sizeof(__half));
    memset(ret, 0, (m + fix_m) * (n + fix_n) * sizeof(__half));
    int ret_cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ret[ret_cnt++] = item[i * n + j];
        }
        for (int j = 0; j < fix_n; j++) {
            ret[ret_cnt++] = 0;
        }
    }
    return ret;
}

int main() {
    int m = 55, n = 37;
    __half *item = (__half *)malloc(m * n * sizeof(__half));
    memset(item, 0, m * n * sizeof(__half));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            item[i * n + j] = static_cast<__half>(static_cast<float>(rand() % 10));
            cout << item[i * n + j] << " ";
        }
        cout << endl;
    }
    item = handle_input(item, m, n);
    cout << item << endl;
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 40; j++) {
            cout << item[i * 40 + j] << " ";
        }
        cout << endl;
    }
}