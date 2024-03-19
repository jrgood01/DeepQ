#include <deque>
#include <mutex>
#include <armadillo>
#include <cstdlib>
#include <ctime> 
#include <vector>
#include <algorithm>

#include "ReplayBuffer.h"
#include "../util/ImageUtil.h"

ReplayBuffer::ReplayBuffer(size_t stateBufferSize, size_t maxBufferSize, size_t batchSize, size_t imageWidth, size_t imageHeight, size_t targetWidth, size_t targetHeight) : 
    maxBufferSize(maxBufferSize), batchSize(batchSize), stateBufferSize(stateBufferSize), imageWidth(imageWidth), imageHeight(imageHeight), 
    targetWidth(targetWidth), targetHeight(targetHeight), curExperience(0), curState(0){
        stateBuffer.resize(stateBufferSize);
        experienceBuffer.reserve(maxBufferSize);

        // Fill the state buffer with empty states
        for (int i = 0; i < stateBufferSize; i++) {
            stateBuffer.at(i) = (std::shared_ptr<unsigned char>(new unsigned char[84 * 84]));
        }
}

/**
 * @brief Adds a state to the replay buffer and the state buffer
 * 
 * @param state State to add
 */
bool ReplayBuffer::addState(Uint32* state, arma::uword bestAction, float reward) {
    if (curExperience >= maxBufferSize) {
        return false;
    }

    std::shared_ptr<unsigned char> processedState  = ProcessImage(
        state, imageWidth, imageHeight, targetWidth, targetHeight);
    arma::mat prevState = getState();
    stateBuffer.at(curState) = processedState;
    if(curExperience > 4) {
        std::shared_ptr<Experience> exp(
            new Experience{prevState, getState(), bestAction, reward}
        );
        if (experienceBuffer.size() >= maxBufferSize) {
            experienceBuffer.erase(experienceBuffer.begin());
        }
        experienceBuffer.push_back(exp);
    }


    curExperience += 1;
    curState = (curState + 1) % stateBufferSize;

    return true;
}



arma::mat ReplayBuffer::getState() {
    arma::mat state = arma::zeros<arma::mat>(84 * 84 * 4, 1);
    for (int i = 0; i < 4; i++) {
        unsigned char* processedImage = stateBuffer.at((curState - i) % stateBufferSize).get();
        for (int j = 0; j < 84 * 84; j++) {
            state(j + i * 84 * 84, 0) = processedImage[j];
        }
    }

    return state;
}

bool ReplayBuffer::readyToTrain() {
    return experienceBuffer.size() > maxBufferSize;
}

int ReplayBuffer::getNextBatchSize() {
    return std::min(static_cast<size_t>(batchSize), experienceBuffer.size());
}

std::vector<std::shared_ptr<Experience>> ReplayBuffer::sampleBatch() {
    // Set random seed based on time
    srand(time(0));
    std::vector<std::shared_ptr<Experience>> batch;
    int max = experienceBuffer.size() > batchSize ? batchSize : experienceBuffer.size();
    for (int i = 0; i < max; i++) {
        int index = rand() % experienceBuffer.size();
        //Copy the experience and bush it to the batch
        Experience* exp = experienceBuffer.at(index).get();
        Experience* newExp = new Experience{exp->state, exp->nextState, exp->action, exp->reward};

        batch.push_back(std::shared_ptr<Experience>(newExp));
    }
    return batch;
}

