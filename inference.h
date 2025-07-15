#pragma once

int predict_mnist_index(int index,
                        float *W1, float *b1,
                        float *W2, float *b2,
                        int input_size, int hidden_size, int output_size,
                        bool verbose = true);
void print_image_ascii(const float *image, int width = 28, int height = 28);