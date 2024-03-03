#include "DQNTrainer.h"
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>
#include <mutex>

#include "Gym.h"
#include "ImageUtil.h"

using namespace mlpack;
using namespace mlpack::ann;
DQNTrainer::DQNTrainer(Gym &gym) : gym(gym), bufferHead(nullptr), bufferTail(nullptr), bufferSize(0){
    // Initialize the model
    model = FFN<>();
    // Main application loop
    // Input size 4x84x84
    // Conv layer 16 8x8 stride 4
    model.Add<mlpack::ann::Convolution>(4, 8, 8, 4, 4, 16);
    // Rectifier nonlinearity
    model.Add<mlpack::ann::ReLU>();
    // 32 4 x 4 stride 2
    model.Add<mlpack::ann::Convolution>(16, 4, 4, 2, 2, 32);
    // Rectifier nonlinearity
    model.Add<mlpack::ann::ReLU>();
    // Fully connected 256
    model.Add<mlpack::ann::Linear>(256);
    model.Add<mlpack::ann::ReLU>();
    // Fully connected size n
    model.Add<mlpack::ann::Linear>(gym.GetNumActions());
    model.Add<mlpack::ann::LogSoftMax>();
    // Set the training parameters
    model.InputDimensions() = std::vector<size_t>({ 84, 84, 4 });

    //Create circular buffer of states
    stateBuffer = new TrainBufferNode(gym.GetState(), 0, 0);
    TrainBufferNode* temp = stateBuffer;
    for (int i = 0; i < 3; i++) {
        stateBuffer->setNext(new TrainBufferNode(gym.GetState(), 0, 0));
        stateBuffer->getNext()->setPrev(stateBuffer);
        stateBuffer = stateBuffer->getNext();
    }
    stateBuffer = stateBuffer->getPrev();
    stateBuffer->setNext(temp);
    temp->setPrev(stateBuffer);

    curState = 0;
}


float DQNTrainer::calculateRollout(int maxDepth, float lambda, TrainBufferNode* node, int curDepth) {
    if (node->getNext() == NULL) {
        std::cout << "Reached end of rollout" << std::endl;
        return -999999;
    }

    float reward = node->getReward();

    float futureReward = 0;
    if (curDepth < maxDepth) {
        futureReward = calculateRollout(maxDepth, lambda, node->getNext(), curDepth + 1);
    }
    return reward + lambda * futureReward;
}

void DQNTrainer::advance() {
    // Update the current state in the circular buffer
    stateBuffer = stateBuffer->getNext();
    stateBuffer->setState(gym.GetState());
    int action = 0;
    if (curState > 4) {
        // Create a matrix that contains the last four states from the circular buffer
        arma::mat state(84 * 84 * 4, 1);
        TrainBufferNode* temp = stateBuffer;
        for (int i = 0; i < 4; i++) {
            unsigned char* processedImage = new unsigned char[84 * 84];
            ProcessImage(temp->getState(), processedImage, 210, 180, 84, 84);
            for (int j = 0; j < 84 * 84; j++) {
                state(j + i * 84 * 84, 0) = processedImage[j];
            }
            delete[] processedImage;
            temp = temp->getPrev();
        }

        // Choose the best action based on the last four states
        action = ChooseBestAction(state);
    } 
    float reward = gym.ApplyAction(action);

    bufferMutex.lock();
    if (bufferHead == nullptr) {
        bufferHead = new TrainBufferNode(gym.GetState(), action, reward);
        bufferTail = bufferHead;
    } else {
        bufferTail->setNext(new TrainBufferNode(gym.GetState(), action, reward));
        bufferTail = bufferTail->getNext();
    }
    bufferSize++;
    bufferMutex.unlock();
    curState += 1;
}

void DQNTrainer::startAdvanceTimer() {
    std::thread([this]() {
        while (true) {
            this->advance();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 200));
        }
    }).detach();
}

void DQNTrainer::deleteBufferHead() {
    if (bufferHead == nullptr) {
        return;
    }
    bufferMutex.lock();
    TrainBufferNode* temp = bufferHead;
    bufferHead = bufferHead->getNext();
    delete temp;
    bufferSize--;
    bufferMutex.unlock();
}

int DQNTrainer::ChooseBestAction(arma::mat& state) {
    float epsilon = 0.1;
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double prob = ((double) rand() / (RAND_MAX));
    std::cout << "Probability: " << prob << std::endl;
    if (prob < epsilon) {
        // Choose a random action
        int action = rand() % gym.GetNumActions();
        std::cout << "Choosing random action: " << action << std::endl;
        return action;
    } else {
        // Reshape the state into the appropriate format for the model
        arma::mat input = arma::vectorise(state);

        // Predict the Q-values for each action
        arma::mat qValues;

        modelMutex.lock();
        model.Predict(input, qValues);
        modelMutex.unlock();

        // Find the action with the highest Q-value
        arma::uword action;
        qValues.max(action);

        // Return the best action
        return action;
    }
}

void DQNTrainer::trainLoop() {
    startAdvanceTimer();
    while (true) {
        if (bufferSize > 100) {
            // Lock the buffer to safely access the training data
            bufferMutex.lock();

            // Initialize input matrix for the model: 4 channels of 84x84 images
            arma::mat input(84 * 84 * 4, 1); // Flattened because MLPack expects column vectors for each sample

            // Assuming bufferHead points to the latest state and we collect 4 subsequent states
            TrainBufferNode* temp = bufferHead;
            for (int i = 0; i < 4; i++) {
                unsigned char* processedImage = new unsigned char[84 * 84]; // Allocate memory for the processed image
                // Process the image from the game environment to the desired format
                ProcessImage(temp->getState(), processedImage, 210, 180, 84, 84);

                // Copy processed image data into the input matrix
                for (int j = 0; j < 84 * 84; j++) {
                    input(j + i * 84 * 84, 0) = processedImage[j];
                }

                // Clean up the processed image memory
                delete[] processedImage;

                // Move to the next state in the buffer
                temp = temp->getNext();
            }

            // Unlock the buffer after accessing the data
            bufferMutex.unlock();

            // Calculate the target value for the current state-action pair
            float rollout = calculateRollout(10, 0.9, temp, 0);

            // Prepare the output vector for training: size should match the number of actions
            arma::mat output(gym.GetNumActions(), 1, arma::fill::zeros);
            output(temp->getAction(), 0) = rollout;

            // Train the model and print the loss and output and expected values
            modelMutex.lock();
            model.Train(input, output);

            if (rollout != 0) {
                //evaluate the model
                arma::mat prediction;
                model.Predict(input, prediction);
                std::cout << "Output: " << prediction << std::endl;
                std::cout << "Expected: " << output << std::endl;
            }

            modelMutex.unlock();

            // Advance the buffer to process new data in the next iteration
            deleteBufferHead();
        }
    }
}

void DQNTrainer::beginTraining() {
    std::thread([this]() {
        trainLoop();
    }).detach();
}