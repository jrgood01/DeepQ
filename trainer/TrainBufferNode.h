#include <SDL.h>
#include <vector>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack.hpp>
using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;
class TrainBufferNode {
public:
    TrainBufferNode(Uint32* state, int action, float reward);
    ~TrainBufferNode();

    void setNext(TrainBufferNode* next);
    void setPrev(TrainBufferNode* prev);
    void setAction(int action);
    void setReward(float reward);
    void setState(Uint32* state);
    void setQvalues(arma::mat qValues);
    arma::mat getQvalues();
    float getReward();
    float getQvalue();
    void setQvalue(float qvalue);
    int getAction();


    TrainBufferNode* getNext();
    TrainBufferNode* getPrev();
    Uint32* getState();


private:
    TrainBufferNode* next;
    TrainBufferNode* prev;
    Uint32* state;

    float reward;
    float curQvalue;
    int action;

    arma::mat qValues;
};