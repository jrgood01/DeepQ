#include "Gym.h"

Gym::Gym() {
}

Gym::~Gym() {
}

void Gym::SetPixel(int x, int y, Uint32 color) {
    if (x >= 0 && y >= 0 && x < screenWidth && y < screenHeight) {
        screenBuffer[y * screenWidth + x] = color;
    }
}

void Gym::DrawRectangle(int x, int y, int rectWidth, int rectHeight, Uint32 color) {
    for (int i = 0; i < rectHeight; i++) {
        for (int j = 0; j < rectWidth; j++) {
            SetPixel(x + j, y + i, color);
        }
    }
}