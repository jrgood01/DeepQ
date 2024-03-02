#ifndef GYM_H
#define GYM_H

#include <string>
#include <SDL.h>

typedef struct Point {
    float x;
    float y;
} Point;

class Gym {
public:
    Gym();
    virtual ~Gym() = 0; // Virtual destructor

    virtual int GetNumActions() = 0;
    virtual std::string ActionToString(int actionID) = 0;
    virtual int ApplyAction(int actionID) = 0;
    virtual Uint32* GetState() = 0;
    virtual Uint16 GetScreenWidth() = 0;
    virtual Uint16 GetScreenHeight() = 0;
protected:
    Uint32* screenBuffer;

    static const Uint16 screenWidth = 210; // Example screen width
    static const Uint16 screenHeight = 160; // Example screen height

    void DrawRectangle(int x, int y, int rectWidth, int rectHeight, Uint32 color);
    void SetPixel(int x, int y, Uint32 color);
};

#endif
