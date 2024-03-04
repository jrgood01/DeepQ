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

    FFN<mlpack::ann::MeanSquaredError, mlpack::ann::RandomInitialization> primaryNet;
    FFN<mlpack::ann::MeanSquaredError, mlpack::ann::RandomInitialization> targetNet;

    TrainBufferNode* bufferHead;
    TrainBufferNode* bufferTail;

    TrainBufferNode* stateBuffer;

    std::mutex bufferMutex;
    std::mutex targetNetMutex;

    //Threadsync binary semaphore

    int curFrame;
    int trainedFrames;
    int framesSinceCopy;
    int bufferSize;

    float calculateRollout(int maxDepth, float lambda, TrainBufferNode* node, int curDepth);
    void startAdvanceTimer();
    void trainLoop();
    void deleteBufferHead();
    arma::mat grabState(TrainBufferNode* node);

    FFN<mlpack::ann::MeanSquaredError, mlpack::ann::RandomInitialization>  createNetwork();
    std::pair<arma::uword, float> ChooseBestAction(arma::mat& state, TrainBufferNode* node);
};