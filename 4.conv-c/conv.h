#include<stdio.h>
#include<stdlib.h>

typedef struct Data {
    int B;
    int C;
    int H;
    int W;
    int *data;
} Data;

typedef struct Core {
    int OUT;
    int C;
    int H;
    int W;
    int *core;
} Core;

void printData(Data *data) {
    for (int i = 0; i < data->B; i++) {
        for (int j = 0; j < data->C; j++) {
            for (int k = 0; k < data->H; k++) {
                for (int v = 0; v < data->W; v++) {
                    printf("%d ", data->data[i * data->C * data->H * data->W + j * data->H * data->W + k * data->W + v]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void printCore(Core *core) {
    for (int i = 0; i < core->C; i++) {
        for (int j = 0; j < core->H; j++) {
            for (int k = 0; k < core->W; k++) {
                printf("%d ", core->core[i * core->H * core->W + j * core->W + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void padData(Data *data, int padding) {
    int h = data->H + padding * 2, w = data->W + padding * 2;
    int *new_data = (int *)malloc(data->B * data->C * h * w * sizeof(int));
    for (int i = 0; i < data->B; i++) {
        for (int j = 0; j < data->C; j++) {
            for (int k = 0; k < padding; k++) {
                for (int v = 0; v < w; v++) {
                    new_data[i * data->C * h * w + j * h * w + k * w + v] = 0;
                }
            }
            for (int k = padding; k < h - padding; k++) {
                for (int v = 0; v < w; v++) {
                    if (v < padding || v >= w - padding) {
                        new_data[i * data->C * h * w + j * h * w + k * w + v] = 0;
                    } else {
                        new_data[i * data->C * h * w + j * h * w + k * w + v] = 
                        data->data[i * data->C * data->H * data->W + j * data->H * data->W + (k - padding) * data->W + v - padding];
                    }
                }
            }
            for (int k = h - padding; k < h; k++) {
                for (int v = 0; v < w; v++) {
                    new_data[i * data->C * h * w + j * h * w + k * w + v] = 0;
                }
            }
        }
    }
    data->data = new_data;
    data->H = h;
    data->W = w;
}