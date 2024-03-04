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
using namespace arma;
using namespace std;
using namespace ens;

DQNTrainer::DQNTrainer(Gym &gym) : gym(gym), bufferHead(nullptr), bufferTail(nullptr), bufferSize(0){
    // Initialize the model
    primaryNet = createNetwork();
    //Target network
    targetNet = createNetwork();

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

    curFrame = 0;
    trainedFrames = 0;
    framesSinceCopy = 0;
}

mlpack::ann::FFN<mlpack::ann::MeanSquaredError, mlpack::ann::RandomInitialization> DQNTrainer::createNetwork() {
    mlpack::ann::FFN<mlpack::ann::MeanSquaredError, mlpack::ann::RandomInitialization> net;
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
    if (curFrame - trainedFrames > 20000) {
        return;
    }
    // Update the current state in the circular buffer
    stateBuffer = stateBuffer->getNext();
    stateBuffer->setState(gym.GetState());
    int action = 0;
    float qVal = 0;
    TrainBufferNode* addNode = new TrainBufferNode(gym.GetState(), 0, 0);
    if (curFrame > 4) {
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
        std::pair<arma::uword, float>  actionAndQVal = ChooseBestAction(state, addNode);
    
        float reward = gym.ApplyAction(actionAndQVal.first);
        addNode->setReward(reward);
        addNode->setQvalue(actionAndQVal.second);

        bufferMutex.lock();
        if (bufferHead == nullptr) {
            bufferHead = addNode;
            bufferTail = bufferHead;
        } else {
            bufferTail->setNext(addNode);
            bufferTail = bufferTail->getNext();
        }
        bufferSize++;
        bufferMutex.unlock();
        // Choose the best action based on the last four states
        curFrame += 1;
    } else {
        gym.ApplyAction(0);
        addNode->setReward(0);
        addNode->setQvalue(0);
        addNode->setQvalues(arma::mat(gym.GetNumActions(), 1, arma::fill::zeros));
    }

    bufferMutex.lock();
    if (bufferHead == nullptr) {
        bufferHead = addNode;
        bufferTail = bufferHead;
    } else {
        bufferTail->setNext(addNode);
        bufferTail = bufferTail->getNext();
    }
    bufferSize++;
    bufferMutex.unlock();
    curFrame += 1;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void DQNTrainer::startAdvanceTimer() {
    std::thread([this]() {
        while (true) {
            this->advance();
        }
    }).detach();
}

void DQNTrainer::deleteBufferHead() {
    std::lock_guard<std::mutex> lock(bufferMutex);
    if (bufferHead == nullptr) {
        return;
    }
    TrainBufferNode* temp = bufferHead;
    bufferHead = bufferHead->getNext();
    if (bufferHead != nullptr) {
        bufferHead->setPrev(nullptr);
    }
    if (bufferHead->getNext() != nullptr) {
        bufferHead->getNext()->setPrev(nullptr);
    }
    temp->setNext(nullptr); // Prevents deleting the rest of the list if delete is called again on temp
    delete temp;
    temp = nullptr; 
    bufferSize--;
}

std::pair<arma::uword, float> DQNTrainer::ChooseBestAction(arma::mat& state, TrainBufferNode* node) {
    float epsilon = 1 - (trainedFrames / 1000000.0);

    if (epsilon < 0.1) {
        epsilon = 0.1;
    }

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double prob = ((double) rand() / (RAND_MAX));
    arma::uword action;
    float  qVal;
    arma::mat input = arma::vectorise(state);

    // Predict the Q-values for each action
    arma::mat qValues;

    targetNetMutex.lock();
    targetNet.Predict(input, qValues);
    targetNetMutex.unlock();
    if (prob < epsilon) {
        // Choose a random action
        action = (arma::uword) rand() % gym.GetNumActions();
        qVal = (float) qValues(action);
    } else {
        qValues.max(action);
        qVal = (float) qValues(action);
    }
    node->setQvalue(qVal);
    node->setAction(action);
    node->setQvalues(qValues);
    
    return std::make_pair(action, qVal);
}

arma::mat DQNTrainer::grabState(TrainBufferNode* node) {
        // Create a matrix that contains the last four states from the circular buffer
        arma::mat state(84 * 84 * 4, 1);
        TrainBufferNode* temp = stateBuffer;
        for (int i = 0; i < 4; i++) {
            if (temp == NULL) {
                return NULL;
            }
            unsigned char* processedImage = new unsigned char[84 * 84];
            ProcessImage(temp->getState(), processedImage, 210, 180, 84, 84);
            for (int j = 0; j < 84 * 84; j++) {
                state(j + i * 84 * 84, 0) = processedImage[j];
            }
            delete[] processedImage;
            temp = temp->getPrev();
        }

        return state;
}


void DQNTrainer::trainLoop() {
    startAdvanceTimer();
    while (trainedFrames < 5000000) {
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
        if (bufferSize >= 512) { // Ensure we have at least a batch's worth of data
            bufferMutex.lock();

            // Initialize matrices for batch inputs and outputs
            arma::mat batchInputs(84 * 84 * 4, 256); // Each column will hold a flattened image
            arma::mat batchOutputs(gym.GetNumActions(), 256, arma::fill::zeros); // Each column corresponds to the output for each image

            // Collect a batch of data
            TrainBufferNode* temp = bufferHead;
            for (int b = 0; b < 256; ++b) {
                // Get the state from the node
                arma::mat state = grabState(temp);

                // Insert the state into the batchInputs matrix
                batchInputs.col(b) = state;

                // Set the corresponding output value for the action taken
                float curQvalue = temp->getQvalue();
                float q_alpha = .5 - (trainedFrames / 1000000.0);
                if (q_alpha < 0.1) {
                    q_alpha = 0.1;
                }
            
    
                //set output to node::qvalues
                arma::mat qValues = temp->getQvalues();
                batchOutputs.col(b) = qValues;
                batchOutputs(temp->getAction(), b) = (1 - q_alpha) * curQvalue + q_alpha * (temp->getReward() + 0.99 * temp->getNext()->getQvalue());

                // Move to the next item in the buffer for the next iteration
                temp = temp->getNext();
            }


            //Shuffle the batch
            arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 255, 256));

            batchInputs = batchInputs.cols(indices);
            batchOutputs = batchOutputs.cols(indices);

            bufferMutex.unlock();

            Adam optimizer(256, 64, 0.9, 0.999, 1e-8, 256, 1e-8, true);


            primaryNet.Train(batchInputs, batchOutputs, optimizer, PrintLoss(), ProgressBar());
            //show outputs of one input
            arma::mat input = batchInputs.col(0);
            arma::mat output;
            primaryNet.Predict(input, output);
            std::cout << "Output: " << output << std::endl;
            std::cout << "Expected: " << batchOutputs.col(0) << std::endl;

            // After training with this batch, delete the processed nodes from the buffer
            for (int i = 0; i < 256; ++i) {
                deleteBufferHead();
            }
            trainedFrames += 256;
            framesSinceCopy += 256;
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