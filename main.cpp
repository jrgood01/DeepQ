#include <iostream>

#include "gui/DisplayWindow.h"
#include "gym/PongGym.h"

int main() {
    // Main application loop
    bool quit = false;
    PongGym myGym = PongGym();
    DisplayWindow gymWindow(myGym.GetScreenWidth(), myGym.GetScreenHeight());
    while (!quit) {
        //Listen for up and down arrow keys
        const Uint8 *state = SDL_GetKeyboardState(NULL);
        gymWindow.UpdateScreenBuffer(myGym.GetState()); // Update the texture with the pixel buffer
        myGym.ApplyAction(0);
        quit = gymWindow.Update(); // Process events and update the window
    }

    return 0;
}