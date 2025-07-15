#pragma once

int predict_mnist_index(int index,
                        float *W1, float *b1,
                        float *W2, float *b2,
                        int input_size, int hidden_size, int output_size,
                        bool verbose = true);
void print_image_ascii(const float *image, int width = 28, int height = 28);

int predict_from_array(float *input,
                       float *W1, float *b1,
                       float *W2, float *b2,
                       int input_size, int hidden_size, int output_size);
void preprocess_image(float *img);
void smooth_image(float *img);
