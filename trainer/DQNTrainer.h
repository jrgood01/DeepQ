// DQNTrainer.h
#include "Gym.h"
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>
using namespace mlpack;
using namespace mlpack::ann;
class DQNTrainer {
public:
    DQNTrainer(Gym &gym);
    // other member functions and variables
private:
    Gym &gym;
    FFN<> model;
    // other member variables
};