#ifndef LAYERS_H
#define LAYERS_H

// Forward propagation of a dense layer: output = W * input + b
void dense_forward(float *input, float *output, float *weights, float *bias, int input_size, int output_size);

// ReLu activation function: max(0, x)
void relu(float *data, int size);

// Softmax function for the final output
void softmax(float *input, float *output, int size);

#endif