//
// Created by Khurram Javed on 2022-01-21.
//

#ifndef INCLUDE_NN_NETWORKS_SNAP1_H_
#define INCLUDE_NN_NETWORKS_SNAP1_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"
#include "../lstm.h"

class Snap1 {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//


  float step_size;
  std::vector<float> predictions;
  std::vector<float> bias;
  std::vector<float> errors;
  float layer_size;

  //  These indexes are used to do parallel computation since std::parallel does not provide rank of a thread
  std::vector<int> indexes;
  std::vector<int> indexes_lstm_cells;

  std::vector<std::vector<float>> prediction_weights;

  std::vector<LinearNeuron> input_neurons;

  std::vector<LSTM> LSTM_neurons;

  std::vector<float> read_output_values();

  Snap1(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features, int layer_size);

  ~Snap1();

  void forward(std::vector<float> inputs);

  float backward(std::vector<float> targets);

  void update_parameters(int layer);

  void reset_state();

};

#endif //INCLUDE_NN_NETWORKS_SNAP1_H_
