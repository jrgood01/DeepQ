// DQNTrainer.h
#include "Gym.h"
#include "TrainBufferNode.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>
#include <mutex>
using namespace mlpack;
using namespace mlpack::ann;
class DQNTrainer {
public:
    DQNTrainer(Gym &gym);
    void advance();
    void beginTraining();
    void pauseTraining();
    void stopTraining();
    void printDimensions();
    // other member functions and variables
private:
    Gym &gym;
    FFN<> model;
    TrainBufferNode* bufferHead;
    TrainBufferNode* bufferTail;

    TrainBufferNode* stateBuffer;
    std::mutex bufferMutex;
    std::mutex modelMutex;
    int curState;
    int bufferSize;
    float calculateRollout(int maxDepth, float lambda, TrainBufferNode* node, int curDepth);
    void startAdvanceTimer();
    void trainLoop();
    void deleteBufferHead();
    int ChooseBestAction(arma::mat& state);
};