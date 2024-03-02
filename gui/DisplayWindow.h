#ifndef DISPLAY_WINDOW_H
#define DISPLAY_WINDOW_H

#include <SDL.h>
class DisplayWindow {
    public:
        DisplayWindow(Uint16 width, Uint16 height);
        void StartDisplay();
        void ClearScreen();
        void UpdateScreenBuffer(Uint32* screenBuffer);
        void InitWindow();
        bool Update();
        void DestroyWindow();

        Uint32* GetScreenBuffer();

    private:
        bool paused;

        Uint16 width;
        Uint16 height;

        SDL_Window* window;
        SDL_Renderer* renderer;
        SDL_Texture* texture;
};

#endif