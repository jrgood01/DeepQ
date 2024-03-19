#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>
#include <mutex>

#include "DQNTrainer.h"
#include "ReplayBuffer.h"
#include "Gym.h"
#include "ImageUtil.h"

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

DQNTrainer::DQNTrainer(Gym &gym) : gym(gym), trainedFrames(0), framesSinceCopy(0), replayBuffer(4, 100000, 256, 210, 160, 84, 84) {
    // Initialize the model
    primaryNet = createNetwork();
    //Target network
    targetNet = createNetwork();

    // Initialize the replay buffer
    Uint32* initialState = gym.GetState();
}

mlpack::ann::FFN<mlpack::ann::MeanSquaredError, mlpack::ann::HeInitialization> DQNTrainer::createNetwork() {
    mlpack::ann::FFN<mlpack::ann::MeanSquaredError, mlpack::ann::HeInitialization> net;
    // Main application loop
    // Input size 4x84x84
    // Conv layer 16 8x8 stride 4
    net.Add<mlpack::ann::Convolution>(4, 8, 8, 4, 4, 16);
    // Rectifier nonlinearity
    net.Add<mlpack::ann::ReLU>();
    // 32 4 x 4 stride 2
    net.Add<mlpack::ann::Convolution>(16, 4, 4, 2, 2, 32);
    // Rectifier nonlinearity
    net.Add<mlpack::ann::ReLU>();
    // Fully connected 256
    net.Add<mlpack::ann::Linear>(256);
    net.Add<mlpack::ann::ReLU>();
    // Fully connected size n
    net.Add<mlpack::ann::Linear>(gym.GetNumActions());
    // Set the training parameters
    net.InputDimensions() = std::vector<size_t>({ 84, 84, 4 });

    return net;
}

void DQNTrainer::advance() {
    // Update the current state in the replay buffer
    Uint32* currentState = gym.GetState();

    // Create a matrix that contains the last four states from the replay buffer
    arma::mat state = replayBuffer.getState();
    // Choose the best action based on the last four states
    std::pair<arma::uword, float> actionAndQVal = ChooseBestAction(state);

    float reward = gym.ApplyAction(actionAndQVal.first);
    // Assuming you have a method to fetch or calculate the next state
    Uint32* nextState = gym.GetState();
    
    replayBuffer.addState(currentState, actionAndQVal.first, reward);

}

std::pair<arma::uword, float> DQNTrainer::ChooseBestAction(arma::mat& state) {
    float epsilon = .9 - (trainedFrames / 1000000.0);
    if (epsilon < 0.1) {
        epsilon = 0.1;
    }
    //epsilon = 0.0;
    double prob = ((double) rand() / (RAND_MAX));
    arma::uword action;
    float qVal;
    arma::mat input = arma::vectorise(state);

    // Predict the Q-values for each action
    arma::mat qValues;

    targetNetMutex.lock();
    primaryNet.Predict(input, qValues);
    targetNetMutex.unlock();

    if (prob < epsilon) {
        // Choose a random action
        action = (arma::uword) rand() % gym.GetNumActions();
        qVal = (float) qValues(action);
    } else {
        qValues.max(action);
        qVal = (float) qValues(action);
    }
    //sleep 10ms
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return std::make_pair(action, qVal);
}

void DQNTrainer::trainLoop() {
    //Try to load the model
    bool success = data::Load("DeepQNet.xml", "model", primaryNet, false);
    if (success) {
        //Copy the primary network to the target network
        targetNet.Parameters() = primaryNet.Parameters();
        std::cout << "File loaded successfully.\n";
    } else {
        std::cout << "Failed to load file.\n";
    }
    while (trainedFrames < 5000000) {
        for (int i = 0; i < 1000; i ++) {
            advance();
        }
        if (framesSinceCopy > 10000 && trainedFrames != 0) {
            std::cout << "Trained frames: " << trainedFrames << std::endl;
            std::cout << "Copying primary network to target network.\n";
            targetNetMutex.lock();
            targetNet.Parameters() = primaryNet.Parameters();
            targetNetMutex.unlock();
            std::cout << "Copy complete.\n";
            framesSinceCopy = 0;
            //save the model
            bool success = data::Save("DeepQNet.xml", "model", primaryNet, false);
            if (success) {
                std::cout << "File saved successfully.\n";
            } else {
                std::cout << "Failed to save file.\n";
            }
        }
        std::cout << "Train size: " << replayBuffer.getNextBatchSize() << "\n";
        for (int i = 0; i < 5; i ++) {
            // Initialize matrices for batch inputs and outputs
            int batchSize = replayBuffer.getNextBatchSize();
            arma::mat batchInputs(84 * 84 * 4, batchSize); // Each column will hold a flattened image
            arma::mat batchOutputs(gym.GetNumActions(), batchSize, arma::fill::zeros); // Each column corresponds to the output for each image

            // Collect a batch of data
            std::vector<std::shared_ptr<Experience>> batch = replayBuffer.sampleBatch();

            // Prepare the batch inputs and outputs
            for (size_t i = 0; i < batch.size(); ++i) {
                batchInputs.col(i) = arma::vectorise(batch[i].get()->state);

                // Predict Q-values for current states with primary network
                arma::mat currentQValues;
                primaryNet.Predict(batch[i].get()->state, currentQValues);

                // Predict Q-values for next states with target network
                arma::mat nextQValues;
                targetNetMutex.lock();
                targetNet.Predict(batch[i].get()->nextState, nextQValues);
                targetNetMutex.unlock();

                // Compute the maximum Q-value for the next states
                double maxNextQValue = arma::max(arma::vectorise(nextQValues));

                // Compute the TD target
                double tdTarget = batch[i].get()->reward;
                tdTarget += 0.99 * maxNextQValue; // Discount factor of 0.99
                // Update the target Q-value for the action taken
                currentQValues(batch[i].get()->action) = tdTarget;

                // Assign the updated Q-values as the target for training
                batchOutputs.col(i) = currentQValues;
            }

            Adam optimizer(batchSize, 64, 0.9, 0.999, 1e-8, 256, 1e-8, true);

            primaryNet.Train(batchInputs, batchOutputs, optimizer, PrintLoss(), ProgressBar());

            //show outputs of one input
            arma::mat input = batchInputs.col(0);
            arma::mat output;
            primaryNet.Predict(input, output);
            std::cout << "Output: " << output << std::endl;
            std::cout << "Expected: " << batchOutputs.col(0) << std::endl;
            trainedFrames += batchSize;
            framesSinceCopy += batchSize;
            std::cout << "Trained frames: " << trainedFrames << " / 1000000" << std::endl;
        }
    }
    std::cout << "Training complete.\n";
}

void DQNTrainer::beginTraining() {
    std::thread([this]() {
        trainLoop();
    }).detach();
}

void DQNTrainer::startAdvanceTimer() {
    std::thread([this]() {
        while (true) {
            this->advance();
        }
    }).detach();
}