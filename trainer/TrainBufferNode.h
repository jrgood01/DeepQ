#include <SDL.h>
#include <vector>

class TrainBufferNode {
public:
    TrainBufferNode(Uint32* state, int action, float reward);
    ~TrainBufferNode();

    void setNext(TrainBufferNode* next);
    void setAction(int action);
    void setReward(float reward);
    float getReward();
    int getAction();

    TrainBufferNode* getNext();
    Uint32* getState();


private:
    TrainBufferNode* next;
    Uint32* state;
    float reward;
    int action;
};