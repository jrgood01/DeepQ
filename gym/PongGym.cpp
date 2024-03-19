#include "PongGym.h"

#include <iostream>
#include <string>
#include <cstdlib> 
#include <cmath>
#include <chrono>
#include <thread>

PongGym::PongGym() : ballLocation{0, 0}, paddleRLocation{0, 0}, paddleLLocation{0, 0}, ballVelocityX(3), ballVelocityY(3) {
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
    return 5;
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

float PongGym::ApplyAction(int actionID){
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
    ballLocation.x += ballVelocityX;
    ballLocation.y += ballVelocityY;
    int collision = DetectCollision();
    float reward = 0.0;
    if (collision == 5) {
        reward = -10.0;
    }
    if (collision == 1 || collision == 2) {
        reward = 10.0;
    }
    UpdateState();
    return reward;
}

/*
Detects collision with padles or walls
*/
int PongGym::DetectCollision() {
    // Get the current time in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    auto epoch = now_ns.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
    long seed = value.count();

    // Use the current time in nanoseconds as the seed for the random number generator
    srand(seed);

    float randomSpeedAdjustment = static_cast<float>(rand() % 20) / 100.0f + .9;

    int collision = 0;
    if (ballLocation.y < 2) {
        ballVelocityY *= -1; // Reverse the y-direction of the ball
        collision = 3;
    }
    if (ballLocation.y > GetScreenHeight() - BALL_HEIGHT) {
        ballVelocityY *= -1; // Reverse the y-direction of the ball
        collision = 4;
    }
    if (ballLocation.x < 2) {
        ballLocation.x = GetScreenWidth() / 2;
        ballLocation.y = GetScreenHeight() / 2;
        collision = 5;
    }
    if (ballLocation.x > GetScreenWidth() - BALL_WIDTH) {
        ballLocation.x = GetScreenWidth() / 2;
        ballLocation.y = GetScreenHeight() / 2;
        collision = 5;
    }
    if (ballLocation.x < paddleLLocation.x + PADDLE_WIDTH && ballLocation.y > paddleLLocation.y && ballLocation.y < paddleLLocation.y + PADDLE_HEIGHT) {
        ballVelocityX *= -1; // Reverse the x-direction of the ball
        collision = 2;
    }
    if (ballLocation.x + BALL_WIDTH > paddleRLocation.x && ballLocation.y > paddleRLocation.y && ballLocation.y < paddleRLocation.y + PADDLE_HEIGHT) {
        ballLocation.x = paddleRLocation.x - BALL_WIDTH;
        ballVelocityX *= -1; // Reverse the x-direction of the ball
        collision = 1;
    }

    if (collision != 0) {
        ballVelocityX *= randomSpeedAdjustment;
        ballVelocityY *= randomSpeedAdjustment;
    }

    if (abs(ballVelocityX) < 0.5) {
        ballVelocityX *= 1.5;
    }
    if (abs(ballVelocityY) < 0.5) {
        ballVelocityY *= 1.5;
    }

    if (abs(ballVelocityX) > 6) {
        ballVelocityX *= 0.5;
    }

    if (abs(ballVelocityY) > 6) {
        ballVelocityY *= 0.5;
    }

    return collision;
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