#include <SDL.h>
#include <vector>

class TrainBufferNode {
public:
    TrainBufferNode(std::vector<Uint32> &state);
    ~TrainBufferNode();
    void setNext(TrainBufferNode* next);

private:
    TrainBufferNode* next;
    std::vector<Uint32> state;
    float reward;
};