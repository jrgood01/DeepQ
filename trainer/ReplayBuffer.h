#ifndef REPLAYBUFFER_H
#define REPLAYBUFFER_H

#include <deque>
#include <mutex>
#include <armadillo>
#include <mlpack/methods/ann/ffn.hpp>
#include <memory>
#include <SDL.h>

struct Experience {
    arma::mat state;
    arma::mat nextState;
    arma::uword action;
    float reward;
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t stateBufferSize, size_t maxBufferSize, size_t batchSize, size_t imageWidth, size_t imageHeight, size_t targetWidth, size_t targetHeight);
    bool addState(Uint32* state, arma::uword bestAction, float reward);
    bool readyToTrain();
    int getNextBatchSize();
    void clear();
    arma::mat getState();
    std::vector<std::shared_ptr<Experience>> sampleBatch();

private:
    size_t maxBufferSize;
    size_t stateBufferSize;

    std::vector<std::shared_ptr<unsigned char>> stateBuffer;
    std::vector<std::shared_ptr<Experience>> experienceBuffer;

    int curExperience;
    int curState;

    int batchSize;
    int imageWidth;
    int imageHeight;

    int targetWidth;
    int targetHeight;

};

#endif // REPLAYBUFFER_H