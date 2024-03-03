#include <SDL.h>
#include <vector>

class TrainBufferNode {
public:
    TrainBufferNode(Uint32* state, int action, float reward);
    ~TrainBufferNode();

    void setNext(TrainBufferNode* next);
    void setPrev(TrainBufferNode* prev);
    void setAction(int action);
    void setReward(float reward);
    void setState(Uint32* state);
    float getReward();
    int getAction();

    TrainBufferNode* getNext();
    TrainBufferNode* getPrev();
    Uint32* getState();


private:
    TrainBufferNode* next;
    TrainBufferNode* prev;
    Uint32* state;
    float reward;
    int action;
};