#include "layers.h"
#include <cmath>
#include <algorithm>

// Output = weights * input + bias
void dense_forward(float *input, float *output, float *weights, float *bias, int input_size, int output_size)
{
    for (int j = 0; j < output_size; ++j)
    {
        output[j] = bias[j];
        for (int i = 0; i < input_size; ++i)
        {
            output[j] += input[i] * weights[j * input_size + i];
        }
    }
}

// ReLU : f(x) = max(0, x)
void relu(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = std::max(0.0f, data[i]);
    }
}

// Softmax : exp(x_i) / sum(exp(x_j))
void softmax(float *input, float *output, int size)
{
    float max_val = input[0];
    for (int i = 1; i < size; ++i)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; ++i)
    {
        output[i] /= sum;
    }
}