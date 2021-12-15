#include<stdio.h>

#define MAX 12

typedef struct Data {
    int C;
    int H;
    int W;
    int data[MAX][MAX];
} Data;

typedef struct Core {
    int H;
    int W;
    int core[MAX][MAX];
} Core;


void conv(Data *data, Core *core) {
    int data_h = data->H, data_w = data->W;
    int core_h = core->H, core_w = core->W;
    
}

int main() {

}