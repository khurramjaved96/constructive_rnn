//
// Created by Khurram Javed on 2022-01-17.
//

#ifndef INCLUDE_NN_NETWORKS_MLP_LSTM_H_
#define INCLUDE_NN_NETWORKS_MLP_LSTM_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "../lstm.h"
#include "./neural_network.h"

class DeepLSTM {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//
  void print_network_state(int layer_no);
  float step_size;
  std::vector<float> predictions;
  std::vector<float> errors;
  std::vector<float> bias;
//  These indexes are used to do parallel computation since std::parallel does not provide rank of a thread
  std::vector<int> indexes;
  std::vector<int> indexes_lstm_cells;


  std::vector<std::vector<std::vector<float>>> prediction_weights;

  std::vector<LinearNeuron> input_neurons;

  std::vector<std::vector<LSTM>> LSTM_neurons;

  std::vector<float> read_output_values();

  DeepLSTM(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features);

  ~DeepLSTM();

  void forward(std::vector<float> inputs, int layer);

  void backward(std::vector<float> targets, int layer);

  void update_parameters(int layer);

  void reset_state();

};


#endif //INCLUDE_NN_NETWORKS_MLP_LSTM_H_
