#include "PongGym.h"

#include <iostream>
#include <string>
#include <cstdlib> 
#include <cmath>

PongGym::PongGym() : ballLocation{0, 0}, paddleRLocation{0, 0}, paddleLLocation{0, 0}, ballVelocityX(1), ballVelocityY(1) {
    paddleRLocation.x = GetScreenWidth() - PADDLE_WIDTH;
    
    ballLocation.x = GetScreenWidth() / 2;
    ballLocation.y = GetScreenHeight() / 2;

    screenBuffer = (Uint32*) malloc(GetScreenWidth() * GetScreenHeight() * sizeof(Uint32));
    memset(screenBuffer, BACKGROUND_COLOR, GetScreenWidth() * GetScreenHeight() * sizeof(Uint32));
    
    std::srand(std::time(nullptr)); // Seed the random number generator once

    UpdateState();
}

int PongGym::GetNumActions() {
    // Four actions: Paddle L up, Paddle L down, Paddle R up, Paddle R down
    return 4;
}

std::string PongGym::ActionToString(int actionID){
    switch(actionID) {
        case (PADDLE_L_UP):
            return "PADDLE_L_UP";
        case (PADDLE_L_DOWN):
            return "PADDLE_L_DOWN";
        case (PADDLE_R_UP):
            return "PADDLE_R_UP";
        case (PADDLE_R_DOWN):
            return "PADDLE_R_DOWN";
        default:
            return "INVALID";
    }
}

void PongGym::UpdateState() {
    memset(screenBuffer, BACKGROUND_COLOR, screenWidth * screenHeight * sizeof(Uint32));
    DrawRectangle(paddleLLocation.x, paddleLLocation.y, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_COLOR);
    DrawRectangle(paddleRLocation.x, paddleRLocation.y, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_COLOR);
    DrawRectangle(ballLocation.x, ballLocation.y, BALL_WIDTH, BALL_WIDTH, BALL_COLOR);
}

int PongGym::ApplyAction(int actionID){
    switch(actionID) {
        case (PADDLE_L_UP):
            if (paddleLLocation.y > 0) {
                paddleLLocation.y -= 10;
            }
            break;
        case (PADDLE_L_DOWN):
            if (paddleLLocation.y < GetScreenHeight() - PADDLE_HEIGHT) {
                paddleLLocation.y += 10;
            }
            break;
        case (PADDLE_R_UP):
            if (paddleRLocation.y > 0) {
                paddleRLocation.y -= 10;
            }
            break; 
        case (PADDLE_R_DOWN):
            if (paddleRLocation.y < GetScreenHeight() - PADDLE_HEIGHT) {
                paddleRLocation.y += 10;
            }
            break; 
    }

    UpdateState();
}

/*
Detects collision with padles or walls
*/
int PongGym::DetectCollision() {
    if (ballLocation.y < 2) {
        ballLocation.y = 0;
        return 3;
    }
    if (ballLocation.y > GetScreenHeight() - BALL_HEIGHT) {
        ballLocation.y = GetScreenHeight() - BALL_HEIGHT;
        return 4;
    }
    if (ballLocation.x < 2) {
        ballLocation.x = 0;
        return 5;
    }
    if (ballLocation.x > GetScreenWidth() - BALL_WIDTH) {
        ballLocation.x = GetScreenWidth() - BALL_WIDTH;
        return 5;
    }
    if (ballLocation.x < paddleLLocation.x + PADDLE_WIDTH && ballLocation.y > paddleLLocation.y && ballLocation.y < paddleLLocation.y + PADDLE_HEIGHT) {
        ballLocation.x = paddleLLocation.x + PADDLE_WIDTH;
        return 2;
    }
    if (ballLocation.x + BALL_WIDTH > paddleRLocation.x && ballLocation.y > paddleRLocation.y && ballLocation.y < paddleRLocation.y + PADDLE_HEIGHT) {
        ballLocation.x = paddleRLocation.x - BALL_WIDTH;
        return 1;
    }
    return 0;
}

Uint16 PongGym::GetScreenWidth() {
    return screenWidth;
}

Uint16 PongGym::GetScreenHeight() {
    return screenHeight;
}

Uint32* PongGym::GetState() {
    return screenBuffer;
}