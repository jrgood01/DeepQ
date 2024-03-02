#ifndef PONG_GYM_H
#define PONG_GYM_H

#define PADDLE_WIDTH 5
#define PADDLE_HEIGHT 40
#define BALL_WIDTH 5
#define BALL_HEIGHT 5
#define BACKGROUND_COLOR 0xFF00000
#define BALL_COLOR 0xFFFFFFFF
#define PADDLE_COLOR 0xFFFFFFFF

#include <utility>
#include "Gym.h"

enum PongGymActions {
    NONE,
    PADDLE_L_UP,
    PADDLE_L_DOWN,
    PADDLE_R_UP,
    PADDLE_R_DOWN
};

class PongGym : Gym {
    public:
        PongGym();

        int GetNumActions();
        int ApplyAction(int actionID);
        std::string ActionToString(int actionID);
        
        Uint32* GetState();
        Uint16 GetScreenWidth();
        Uint16 GetScreenHeight();

        void UpdateState();

    private:
        Point ballLocation;
        Point paddleRLocation;
        Point paddleLLocation;

        float ballVelocityX;
        float ballVelocityY;
        
        int DetectCollision();

        bool IsCollisionRightPaddle();
        bool IsCollisionLeftPaddle();
};

#endif