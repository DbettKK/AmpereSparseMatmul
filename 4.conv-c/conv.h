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

typedef struct Output {
    int B;
    int OUT;
    int H, W;
    int *output;
} Output;

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
    for (int i = 0; i < core->OUT; i++) {
        for (int j = 0; j < core->C; j++) {
            for (int k = 0; k < core->H; k++) {
                for (int v = 0; v < core->W; v++) {
                    printf("%d ", core->core[i * core->C * core->H * core->W + j * core->H * core->W + k * core->W + v]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void printOutput(Output *out) {
    for (int i = 0; i < out->B; i++) {
        for (int j = 0; j < out->OUT; j++) {
            for (int k = 0; k < out->H; k++) {
                for (int v = 0; v < out->W; v++) {
                    printf("%d ", out->output[i * out->OUT * out->H * out->W + j * out->H * out->W + k * out->W + v]);
                }
                printf("\n");
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

int isXInData(Data *data, int x) {
    return x >= 0 && x < data->H;
}

int isYInData(Data *data, int y) {
    return y >= 0 && y < data->W;
}

void conv(Data *data, Core *core, int stride, Output *out) {
    for (int i = 0; i < data->B; i++) {
        for (int j = 0; j < core->OUT; j++) {
            int out_x = 0, out_y = 0;
            while(isXInData(data, out_x * stride + core->H - 1)) {
                int total = 0;
                for (int k = 0; k < data->C; k++) {
                    for (int c_x = 0; c_x < core->H; c_x++) {
                        for (int c_y = 0; c_y < core->W; c_y++) {
                            int d_x = c_x + out_x * stride;
                            int d_y = c_y + out_y * stride;
                            total += data->data[i * data->C * data->H * data->W + k * data->H * data->W + d_x * data->W + d_y] * 
                                core->core[j * core->C * core->H * core->W + k * core->H * core->W + c_x * core->W + c_y];
                        }
                    }
                }
                out->output[i * out->OUT * out->H * out->W + j * out->H * out->W + out_x * out->W + out_y] = total;
                out_y++;
                if (!isYInData(data, out_y * stride + core->W - 1)) {
                    out_y = 0;
                    out_x++;
                }
            }
        }
    }
}

void generateData(Data *data) {
    int cnt = 1;
    for (int i = 0; i < data->B; i++) {
        for (int j = 0; j < data->C; j++) {
            for (int k = 0; k < data->H; k++) {
                for (int v = 0; v < data->W; v++) {
                    data->data[i * data->C * data->H * data->W + j * data->H * data->W + k * data->W + v] = cnt++;
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void generateCore(Core *core) {
    for (int i = 0; i < core->OUT; i++) {
        for (int j = 0; j < core->C; j++) {
            for (int k = 0; k < core->H; k++) {
                for (int v = 0; v < core->W; v++) {
                    core->core[i * core->C * core->H * core->W + j * core->H * core->W + k * core->W + v] = 1;
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}
