#include "model.h"
#include <fstream>

void save_model(const float *W1, const float *b1,
                const float *W2, const float *b2,
                int in_size, int hidden_size, int out_size,
                const char *filename)
{
    std::ofstream out(filename, std::ios::binary);
    out.write((char *)W1, sizeof(float) * in_size * hidden_size);
    out.write((char *)b1, sizeof(float) * hidden_size);
    out.write((char *)W2, sizeof(float) * hidden_size * out_size);
    out.write((char *)b2, sizeof(float) * out_size);
}

bool load_model(float *W1, float *b1,
                float *W2, float *b2,
                int in_size, int hidden_size, int out_size,
                const char *filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        return false;
    in.read((char *)W1, sizeof(float) * in_size * hidden_size);
    in.read((char *)b1, sizeof(float) * hidden_size);
    in.read((char *)W2, sizeof(float) * hidden_size * out_size);
    in.read((char *)b2, sizeof(float) * out_size);
    return true;
}
