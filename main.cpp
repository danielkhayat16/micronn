#include <iostream>
#include <ctime>
#include <cstdlib>
#include "layers.h"
#include "mnist_loader.h"
#include "train.h"
#include <cmath>

#define INPUT_SIZE 784  // 28 x 28 image
#define HIDDEN_SIZE 128 // number of hiden neurones
#define OUTPUT_SIZE 10  // 10 classes (digits from 0 to 9)
#define TRAIN_SIZE 6000
#define EPOCHS 5
float evaluate_accuracy(float *W1, float *b1,
                        float *W2, float *b2,
                        int input_size, int hidden_size, int output_size);

int main()
{
    srand(time(NULL));
    float W1[HIDDEN_SIZE * INPUT_SIZE];
    float b1[HIDDEN_SIZE];
    float W2[HIDDEN_SIZE * OUTPUT_SIZE];
    float b2[OUTPUT_SIZE];

    // Allows to initialize Xavier weights
    float limit1 = std::sqrt(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float limit2 = std::sqrt(6.0f / (HIDDEN_SIZE + OUTPUT_SIZE));

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; ++i)
        W1[i] = ((float)rand() / RAND_MAX * 2 - 1) * limit1;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        b1[i] = 0.0f;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; ++i)
        W2[i] = ((float)rand() / RAND_MAX * 2 - 1) * limit2;
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        b2[i] = 0.0f;

    float input[INPUT_SIZE];
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        std::cout << "\n🔁 Epoch " << (epoch + 1) << " / " << EPOCHS << "\n";

        float total_loss = 0.0f;

        for (int step = 0; step < TRAIN_SIZE * 10; ++step)
        {
            int index = rand() % TRAIN_SIZE;
            int label = load_mnist_label("data/train-labels-idx1-ubyte", index);
            if (!load_mnist_image("data/train-images-idx3-ubyte", index, input))
            {
                std::cerr << "Erreur lecture image\n";
                continue;
            }

            float loss = train_one_step(input, label, W1, b1, W2, b2, 0.001f, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            total_loss += loss;
            if (step % 10000 == 0)
                std::cout << "Step " << step << " - Loss: " << loss << "\n";

            if (step % 10000 == 0)
                std::cout << "Step " << step << " OK\n";
        }
        float avg_loss = total_loss / (TRAIN_SIZE * 10);
        std::cout << "📉 Moyenne loss (epoch " << (epoch + 1) << ") : " << avg_loss << "\n";

        evaluate_accuracy(W1, b1, W2, b2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }

    std::cout << "Entraînement terminé ✅\n";

    return 0;
}

float evaluate_accuracy(float *W1, float *b1,
                        float *W2, float *b2,
                        int input_size, int hidden_size, int output_size)
{

    int correct = 0;
    float input[784];
    float hidden[512];
    float output[128];
    float probs[128];

    for (int i = 0; i < 1000; ++i)
    {
        if (!load_mnist_image("data/t10k-images-idx3-ubyte", i, input))
            continue;
        int label = load_mnist_label("data/t10k-labels-idx1-ubyte", i);

        dense_forward(input, hidden, W1, b1, input_size, hidden_size);
        relu(hidden, hidden_size);
        dense_forward(hidden, output, W2, b2, hidden_size, output_size);
        softmax(output, probs, output_size);

        int predicted = 0;
        float max_prob = probs[0];
        for (int j = 1; j < output_size; ++j)
        {
            if (probs[j] > max_prob)
            {
                max_prob = probs[j];
                predicted = j;
            }
        }

        if (predicted == label)
            ++correct;
    }

    float accuracy = 100.0f * correct / 1000.0f;
    std::cout << "✔️ Accuracy: " << correct << " / 1000 = " << accuracy << "%\n";
    return accuracy;
}