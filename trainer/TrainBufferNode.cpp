#include <SDL.h>
#include <vector>
#include "TrainBufferNode.h"

TrainBufferNode::TrainBufferNode(Uint32* state, int action, float reward): action(action), reward(reward){
    //Deep copy the state
    this->state = (Uint32*)malloc(210 * 160 * sizeof(Uint32));
    this->state = (Uint32*)memcpy(this->state, state, 210 * 160 * sizeof(Uint32));
}

TrainBufferNode::~TrainBufferNode() {
    delete[] state;
}

void TrainBufferNode::setNext(TrainBufferNode* next) {
    this->next = next;
}

TrainBufferNode* TrainBufferNode::getNext() {
    return next;
}