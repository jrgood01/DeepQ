// DQNTrainer.h
#include "Gym.h"
#include "TrainBufferNode.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>

using namespace mlpack;
using namespace mlpack::ann;
class DQNTrainer {
public:
    DQNTrainer(Gym &gym);
    void advance();
    // other member functions and variables
private:
    Gym &gym;
    FFN<> model;
    TrainBufferNode* bufferHead;
    TrainBufferNode* bufferTail;
    int bufferSize;
    // other member variables
};