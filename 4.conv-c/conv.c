#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include"conv.h"

void input() {
    Data *data = malloc(sizeof(Data));
    scanf("%d%d%d%d", &data->B, &data->C, &data->H, &data->W);
    int data_size = data->B * data->C * data->H * data->W;
    data->data = malloc(data_size * sizeof(int));
    
    Core *core = malloc(sizeof(Core));
    scanf("%d%d%d", &core->OUT, &core->H, &core->W);
    core->C = data->C;
    int core_size = core->OUT * core->C * core->H * core->W;
    core->core = malloc(core_size * sizeof(int));

    int padding, stride;
    scanf("%d%d", &padding, &stride);

    generateData(data);
    generateCore(core);

    padData(data, padding);

    printData(data);
    printCore(core);
    
    Output *out = malloc(sizeof(Output));
    out->B = data->B;
    out->OUT = core->OUT;
    out->H = (data->H - core->H) / stride + 1;  // 输出矩阵的高
    out->W = (data->W - core->W) / stride + 1;  // 输出矩阵的宽
    out->output = malloc(out->B * out->OUT * out->H * out->W * sizeof(int));

    conv(data, core, stride, out);

    printOutput(out);
}


int main() {
    input();
    return 0;
}