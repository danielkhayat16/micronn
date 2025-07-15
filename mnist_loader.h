#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
bool load_mnist_image(const std::string& image_filename, int index, float* out_image);
int load_mnist_label(const std::string& label_filename, int index);

#endif
