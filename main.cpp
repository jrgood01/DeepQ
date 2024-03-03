#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>

#include "gui/DisplayWindow.h"
#include "gym/PongGym.h"
#include "trainer/DQNTrainer.h"
#include "util/ImageUtil.h"


using namespace mlpack;
using namespace mlpack::ann;
int main() {
    bool quit = false;
    PongGym myGym = PongGym();
    DQNTrainer trainer = DQNTrainer(myGym);
    trainer.beginTraining();
    DisplayWindow gymWindow(myGym.GetScreenWidth(), myGym.GetScreenHeight());
    while (!quit) {
        //Listen for up and down arrow keys
        const Uint8 *state = SDL_GetKeyboardState(NULL);
        gymWindow.UpdateScreenBuffer(myGym.GetState()); // Update the texture with the pixel buffer
        
        quit = gymWindow.Update(); // Process events and update the window
    }

    return 0;
}