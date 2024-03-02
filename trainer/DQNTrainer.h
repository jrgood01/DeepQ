#define PROTO_PATH "model/DQN_Proto.txt"
#define WEIGHTS_PATH "model/DQN_Weights.caffemodel"

#include <memory>
class DQNTrainer {
    public:
        DQNTrainer(int numberOutput);
        std::shared_ptr<std::vector<float>> Inference(const std::vector<float>& input);
    private:
        static std::string modifyNetworkConfig(const std::string& filePath, int numberOutput);
};