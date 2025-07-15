#ifndef TRAIN_H
#define TRAIN_H

float train_one_step(float *input, int label,
                     float *W1, float *b1,
                     float *W2, float *b2,
                     float learning_rate,
                     int input_size,
                     int hidden_size,
                     int output_size);

#endif
