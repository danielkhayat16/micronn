#pragma once

void save_model(const float *W1, const float *b1,
                const float *W2, const float *b2,
                int in_size, int hidden_size, int out_size,
                const char *filename);

bool load_model(float *W1, float *b1,
                float *W2, float *b2,
                int in_size, int hidden_size, int out_size,
                const char *filename);
