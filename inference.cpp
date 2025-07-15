#include "inference.h"
#include "layers.h"
#include "mnist_loader.h"
#include <iostream>
#include <cmath>
#include <cstring>

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

int predict_from_array(float *input, float *W1, float *b1, float *W2, float *b2,
                       int input_size, int hidden_size, int output_size)
{
    float hidden[hidden_size];
    float output[output_size];
    float probs[output_size];

    dense_forward(input, hidden, W1, b1, input_size, hidden_size);
    relu(hidden, hidden_size);

    dense_forward(hidden, output, W2, b2, hidden_size, output_size);
    softmax(output, probs, output_size);

    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 0; i < output_size; ++i)
    {
        std::cout << "Prob[" << i << "] = " << probs[i] << "\n";
    }

    for (int i = 1; i < output_size; ++i)
    {
        if (probs[i] > max_prob)
        {
            max_prob = probs[i];
            predicted = i;
        }
    }
    return predicted;
}

void preprocess_image(float *img)
{
    int sum_x = 0, sum_y = 0, count = 0;

    // Calcul du barycentre des pixels "non vides"
    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            float val = img[y * 28 + x];
            if (val > 0.1f)
            { // pixel considéré comme dessiné
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }

    if (count == 0)
        return; // rien à centrer

    int center_x = sum_x / count;
    int center_y = sum_y / count;

    int dx = 14 - center_x;
    int dy = 14 - center_y;

    float shifted[784] = {0.0f};

    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28)
            {
                shifted[ny * 28 + nx] = img[y * 28 + x];
            }
        }
    }

    std::memcpy(img, shifted, 784 * sizeof(float));
}

void smooth_image(float *img)
{
    float temp[784] = {0.0f};
    for (int y = 1; y < 27; ++y)
    {
        for (int x = 1; x < 27; ++x)
        {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    sum += img[(y + dy) * 28 + (x + dx)];
            temp[y * 28 + x] = sum / 9.0f; // moyenne des voisins
        }
    }
    std::memcpy(img, temp, 784 * sizeof(float));
}
