#include <SDL.h>
#include <vector>
#include "TrainBuffer.h"

TrainBufferNode::TrainBufferNode(std::vector<Uint32> &state) {
    this->state = std::vector<Uint32>(state);
}