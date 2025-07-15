#include "inference.h"
#include "layers.h"
#include "mnist_loader.h"
#include <iostream>

int predict_mnist_index(int index,
                        float *W1, float *b1,
                        float *W2, float *b2,
                        int input_size, int hidden_size, int output_size,
                        bool verbose)
{
    float input[784], hidden[512], output[128], probs[128];
    if (!load_mnist_image("data/t10k-images-idx3-ubyte", index, input))
    {
        std::cerr << "Erreur while loading mnist image\n";
        return -1;
    }
    int label = load_mnist_label("data/t10k-labels-idx1-ubyte", index);
    dense_forward(input, hidden, W1, b1, input_size, hidden_size);
    relu(hidden, hidden_size);
    dense_forward(hidden, output, W2, b2, hidden_size, output_size);
    softmax(output, probs, output_size);

    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 0; i < output_size; ++i)
    {
        if (probs[i] > max_prob)
        {
            max_prob = probs[i];
            predicted = i;
        }
    }
    if (verbose)
    {
        print_image_ascii(input);
        std::cout << "\n🎯 Prediction sur image #" << index << " :\n";
        std::cout << "  ✅ Vrai chiffre     : " << label << "\n";
        std::cout << "  🤖 Chiffre prédit  : " << predicted << "\n";
        std::cout << "  📊 Probabilité     : " << max_prob << "\n";
    }

    return predicted;
}

void print_image_ascii(const float *image, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float pixel = image[y * width + x];
            if (pixel > 0.75f)
                std::cout << "█";
            else if (pixel > 0.5f)
                std::cout << "▓";
            else if (pixel > 0.25f)
                std::cout << "▒";
            else if (pixel > 0.1f)
                std::cout << "░";
            else
                std::cout << " ";
        }
        std::cout << "\n";
    }
}