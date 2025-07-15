#include "train.h"
#include "layers.h"
#include <cmath>
#include <cstring>
#include <iostream>

float hidden[512];
float relu_grad[512];
float output[128];
float probs[128];
float grad_output[128];
float grad_hidden[512];

float train_one_step(float *input, int label, float *W1, float *b1,
                     float *W2, float *b2, float learning_rate, int input_size,
                     int hidden_size, int output_size)
{

    // Forward pass
    dense_forward(input, hidden, W1, b1, input_size, hidden_size);
    relu(hidden, hidden_size);
    dense_forward(hidden, output, W2, b2, hidden_size, output_size);
    softmax(output, probs, output_size);

    // Gradients: softmax + cross-entropy
    for (int i = 0; i < output_size; ++i)
    {
        grad_output[i] = probs[i] - (i == label ? 1.0f : 0.0f);
    }

    // Backprop layer W2
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < hidden_size; ++j)
        {
            W2[i * hidden_size + j] -= learning_rate * grad_output[i] * hidden[j];
        }
        b2[i] -= learning_rate * grad_output[i];
    }

    // Backprop to the hidden layer
    for (int j = 0; j < hidden_size; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < output_size; ++i)
        {
            sum += grad_output[i] * W2[i * hidden_size + j];
        }
        grad_hidden[j] = (hidden[j] > 0.0f) ? sum : 0.0f;
    }

    // Backprop layer W1
    for (int j = 0; j < hidden_size; ++j)
    {
        for (int i = 0; i < input_size; ++i)
        {
            W1[j * input_size + i] -= learning_rate * grad_hidden[j] * input[i];
        }
        b1[j] -= learning_rate * grad_hidden[j];
    }

    float loss = -std::log(probs[label] + 1e-9f);
    return loss;
}
