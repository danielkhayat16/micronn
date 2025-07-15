#include <SDL2/SDL.h>
#include <iostream>
#include <cstring>
#include "model.h"
#include "inference.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

const int GRID_SIZE = 28;
const int PIXEL_SIZE = 10;
const int WINDOW_SIZE = GRID_SIZE * PIXEL_SIZE;

float W1[HIDDEN_SIZE * INPUT_SIZE];
float b1[HIDDEN_SIZE];
float W2[OUTPUT_SIZE * HIDDEN_SIZE];
float b2[OUTPUT_SIZE];

float image[GRID_SIZE * GRID_SIZE] = {0};

void clear_image()
{
    std::memset(image, 0, sizeof(image));
}

void draw_grid(SDL_Renderer *renderer)
{
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    for (int i = 0; i <= GRID_SIZE; ++i)
    {
        SDL_RenderDrawLine(renderer, i * PIXEL_SIZE, 0, i * PIXEL_SIZE, WINDOW_SIZE);
        SDL_RenderDrawLine(renderer, 0, i * PIXEL_SIZE, WINDOW_SIZE, i * PIXEL_SIZE);
    }
}

void draw_image(SDL_Renderer *renderer)
{
    for (int y = 0; y < GRID_SIZE; ++y)
    {
        for (int x = 0; x < GRID_SIZE; ++x)
        {
            float val = image[y * GRID_SIZE + x];
            if (val > 0.0f)
            {
                int shade = static_cast<int>(val * 255);
                SDL_SetRenderDrawColor(renderer, shade, shade, shade, 255);
                SDL_Rect rect = {x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE};
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
}
void print_ascii_image()
{
    for (int y = 0; y < GRID_SIZE; ++y)
    {
        for (int x = 0; x < GRID_SIZE; ++x)
        {
            float val = image[y * GRID_SIZE + x];
            std::cout << (val > 0.5f ? "#" : (val > 0.1f ? "." : " "));
        }
        std::cout << "\n";
    }
}

int main(int argc, char *argv[])
{
    const char *model_file = "model.bin";
    if (!load_model(W1, b1, W2, b2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, model_file))
    {
        std::cerr << "Erreur chargement du modèle.\n";
        return 1;
    }
    else
    {
        std::cout << "📦 Modèle chargé depuis model.bin\n";
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "SDL init error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Draw a digit", 100, 100, WINDOW_SIZE, WINDOW_SIZE, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    bool running = true;
    bool drawing = false;
    SDL_Event e;

    while (running)
    {
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
            {
                running = false;
            }
            else if (e.type == SDL_MOUSEBUTTONDOWN)
            {
                drawing = true;
            }
            else if (e.type == SDL_MOUSEBUTTONUP)
            {
                drawing = false;
            }
            else if (e.type == SDL_KEYDOWN)
            {
                if (e.key.keysym.sym == SDLK_c)
                {
                    clear_image();
                }
                else if (e.key.keysym.sym == SDLK_SPACE)
                {
                    smooth_image(image);
                    preprocess_image(image);
                    int predicted = predict_from_array(image, W1, b1, W2, b2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
                    std::cout << "🧠 Prédiction du modèle : " << predicted << std::endl;
                }
            }
            else if (drawing && (e.type == SDL_MOUSEMOTION || e.type == SDL_MOUSEBUTTONDOWN))
            {
                int x, y;
                SDL_GetMouseState(&x, &y);
                int gx = x / PIXEL_SIZE;
                int gy = y / PIXEL_SIZE;
                if (gx > 0)
                    image[gy * GRID_SIZE + gx - 1] = std::max(image[gy * GRID_SIZE + gx - 1], 0.5f);
                if (gx < GRID_SIZE - 1)
                    image[gy * GRID_SIZE + gx + 1] = std::max(image[gy * GRID_SIZE + gx + 1], 0.5f);
                if (gy > 0)
                    image[(gy - 1) * GRID_SIZE + gx] = std::max(image[(gy - 1) * GRID_SIZE + gx], 0.5f);
                if (gy < GRID_SIZE - 1)
                    image[(gy + 1) * GRID_SIZE + gx] = std::max(image[(gy + 1) * GRID_SIZE + gx], 0.5f);

                if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE)
                {
                    image[gy * GRID_SIZE + gx] = 1.0f;
                }
            }
        }
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        draw_image(renderer);
        draw_grid(renderer);
        SDL_RenderPresent(renderer);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}