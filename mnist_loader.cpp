#include "mnist_loader.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

// Util function to read  uint32 big endian
static uint32_t read_be_uint32(std::ifstream &f)
{
    uint32_t x;
    f.read(reinterpret_cast<char *>(&x), 4);
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24);
}

bool load_mnist_image(const std::string &image_filename, int index, float *out_image)
{
    std::ifstream f(image_filename, std::ios::binary);
    if (!f)
        return false;

    uint32_t magic = read_be_uint32(f);
    uint32_t n_images = read_be_uint32(f);
    uint32_t rows = read_be_uint32(f);
    uint32_t cols = read_be_uint32(f);
    if (magic != 0x00000803)
        return false;
    if (index < 0 || index >= (int)n_images)
        return false;

    size_t image_size = rows * cols;
    f.seekg(16 + index * image_size);
    std::vector<uint8_t> buffer(image_size);
    f.read(reinterpret_cast<char *>(buffer.data()), image_size);
    for (size_t i = 0; i < image_size; ++i)
        out_image[i] = buffer[i] / 255.0f;

    return true;
}

int load_mnist_label(const std::string &label_filename, int index)
{
    std::ifstream f(label_filename, std::ios::binary);
    if (!f)
        return -1;

    uint32_t magic = read_be_uint32(f);
    uint32_t n_labels = read_be_uint32(f);
    if (magic != 0x00000801)
        return -1;
    if (index < 0 || index >= (int)n_labels)
        return -1;

    f.seekg(8 + index);
    uint8_t lbl;
    f.read(reinterpret_cast<char *>(&lbl), 1);
    return static_cast<int>(lbl);
}