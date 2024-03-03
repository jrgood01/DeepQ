#include <SDL.h>
#include <vector>
#include "TrainBufferNode.h"
#include <iostream>

TrainBufferNode::TrainBufferNode(Uint32* state, int action, float reward): action(action), reward(reward) {
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

void TrainBufferNode::setState(Uint32* state) {
    this->state = (Uint32*)memcpy(this->state, state, 210 * 160 * sizeof(Uint32));
}

void TrainBufferNode::setPrev(TrainBufferNode* prev) {
    this->prev = prev;
}

TrainBufferNode* TrainBufferNode::getPrev() {
    return prev;
}

TrainBufferNode* TrainBufferNode::getNext() {
    return next;
}

void TrainBufferNode::setAction(int action) {
    this->action = action;
}

int TrainBufferNode::getAction() {
    return action;
}

void TrainBufferNode::setReward(float reward) {
    this->reward = reward;
}

float TrainBufferNode::getReward() {
    return reward;
}

Uint32* TrainBufferNode::getState() {
    return state;
}
