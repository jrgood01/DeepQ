#include "DQNTrainer.h"
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>
#include "Gym.h"
using namespace mlpack;
using namespace mlpack::ann;
DQNTrainer::DQNTrainer(Gym &gym) : gym(gym) {
    // Main application loop
    FFN<> model;
    // Input size 84x84x4
    // Conv layer 16 8x8 stride 4
    model.Add<Convolution>(16, 8, 8, 4, 4);
    // Rectifier nonlinearity
    model.Add<ReLU>();
    // 32 4 x 4 stride 2
    model.Add<Convolution>(32, 4, 4, 2, 2);
    // Rectifier nonlinearity
    model.Add<ReLU>();
    // Fully connected 256
    model.Add<Linear>(256);
    model.Add<ReLU>();
    // Fully connected size n
    model.Add<Linear>(gym.GetNumActions());
    model.Add<LogSoftMax>();
    // Set the training parameters
}

void DQNTrainer::advance() {
    float reward = gym.ApplyAction(0);
    if (bufferHead == NULL) {
        bufferHead = new TrainBufferNode(gym.GetState(), 0, reward);
        bufferTail = bufferHead;
    } else {
        bufferTail->setNext(new TrainBufferNode(gym.GetState(), 0, reward));
        bufferTail = bufferTail->getNext();
    }
    bufferSize++;
    std::cout << "Buffer size: " << bufferSize << " Bytes: " << bufferSize * 210 * 180 << std::endl;
}
