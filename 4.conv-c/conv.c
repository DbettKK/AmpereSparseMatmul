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
    scanf("%d%d%d", &core->C, &core->H, &core->W);
    int core_size = core->C * core->H * core->W;
    core->core = malloc(core_size * sizeof(int));

    int padding, stride;
    scanf("%d%d", &padding, &stride);

    printData(data);
    padData(data, padding);
    printData(data);
    //conv(data, core);
}

void conv(Data *data, Core *core) {
    int data_h = data->H, data_w = data->W;
    int core_h = core->H, core_w = core->W;
    
}

int main() {
    input();
    return 0;
}