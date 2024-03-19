// DQNTrainer.h
#include "Gym.h"
#include "TrainBufferNode.h"
#include "ReplayBuffer.h"

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

    FFN<mlpack::ann::MeanSquaredError, mlpack::ann::HeInitialization> primaryNet;
    FFN<mlpack::ann::MeanSquaredError, mlpack::ann::HeInitialization> targetNet;

    ReplayBuffer replayBuffer;

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
    arma::mat grabState();
    void addNodeToBuffer(TrainBufferNode* node);
    void collectBatch(arma::mat& batchInputs, arma::mat& batchOutputs);

    FFN<mlpack::ann::MeanSquaredError, mlpack::ann::HeInitialization>  createNetwork();
    std::pair<arma::uword, float> ChooseBestAction(arma::mat& state);
};