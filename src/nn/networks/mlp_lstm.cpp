//
// Created by Khurram Javed on 2022-01-17.
//


#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/mlp_lstm.h"
#include "../../../include/nn/neuron.h"
#include "../../../include/nn/lstm.h"


void DeepLSTM::print_network_state(int layer_no) {
  std::cout << "Layer\tID\tValue\tIncoming\n";
  for(int layer = 0; layer < layer_no+1; layer++){
    for(int neuron = 0; neuron < LSTM_neurons[layer].size(); neuron++){
      std::cout << layer << "\t" << LSTM_neurons[layer][neuron].id << "\t" << LSTM_neurons[layer][neuron].value << "\t" << LSTM_neurons[layer][neuron].incoming_neurons.size() << "\n";
    }
  }
}
DeepLSTM::DeepLSTM(float step_size,
                   int seed,
                   int no_of_input_features,
                   int total_targets,
                   int total_recurrent_features) {
  this->step_size = step_size;
  this->mt.seed(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron n(true, false);
    this->input_neurons.push_back(n);
  }

  for (int i = 0; i < total_recurrent_features; i++) {
    indexes_lstm_cells.push_back(i);
  }
//
  for (int layer = 0; layer < 100; layer++) {
    std::vector<LSTM> lstm_layer;
    for (int i = 0; i < total_recurrent_features; i++) {
//    std::cout << "Recurrent feature no "<< i << std::endl;
      LSTM lstm_neuron(weight_sampler(mt),
                       weight_sampler(mt),
                       weight_sampler(mt),
                       weight_sampler(mt),
                       weight_sampler(mt),
                       weight_sampler(mt),
                       weight_sampler(mt),
                       weight_sampler(mt)
                       ,0.01);
//      if(layer == 0) {
        for (int counter = 0; counter < this->input_neurons.size(); counter++) {
          Neuron *neuron_ref = &this->input_neurons[counter];
          lstm_neuron.add_synapse(neuron_ref,
                                  weight_sampler(mt),
                                  weight_sampler(mt),
                                  weight_sampler(mt),
                                  weight_sampler(mt));
        }
//      }
      if(layer > 0) {
        for (int layer_temp = 0; layer_temp < layer; layer_temp++) {
          for (int neuron_temp = 0; neuron_temp < this->LSTM_neurons[layer_temp].size(); neuron_temp++) {
            Neuron *neuron_ref = &this->LSTM_neurons[layer_temp][neuron_temp];
//          std::cout << "Adding LSTM as input\n";
            lstm_neuron.add_synapse(neuron_ref,
                                    weight_sampler(mt),
                                    weight_sampler(mt),
                                    weight_sampler(mt),
                                    weight_sampler(mt));
          }
        }
      }

      lstm_layer.push_back(lstm_neuron);
    }
    this->LSTM_neurons.push_back(lstm_layer);
  }
//  exit(1);

  for (int i = 0; i < total_targets; i++) {
    predictions.push_back(0);
    bias.push_back(0);
    errors.push_back(0);
    indexes.push_back(i);
    std::vector<std::vector<float>> prediction_weight_cur_target;
    for (int j = 0; j < this->LSTM_neurons.size(); j++) {
      std::vector<float> inner_most_weights;
      for (int inner_c = 0; inner_c < this->LSTM_neurons[j].size(); inner_c++) {
        inner_most_weights.push_back(0);
      }
      prediction_weight_cur_target.push_back(inner_most_weights);
    }
    prediction_weights.push_back(prediction_weight_cur_target);
  }
}

void DeepLSTM::reset_state() {
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    for (int j = 0; j < LSTM_neurons[i].size(); j++) {
      LSTM_neurons[i][j].reset_state();
    }
  }
}

void DeepLSTM::forward(std::vector<float> inputs, int layer) {
  this->time_step++;
//  Set input neurons value
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = inputs[i];
  }
  for (int i = 0; i < layer+1; i++) {
    std::for_each(
        std::execution::par_unseq,
        this->LSTM_neurons[i].begin(),
        this->LSTM_neurons[i].end(),
        [&](LSTM &n) {
          n.update_value_sync();
          if (i == layer)
            n.compute_gradient_of_all_synapses();
        });
  }


  for (int i = 0; i < layer+1; i++) {
    std::for_each(
        std::execution::par_unseq,
        this->LSTM_neurons[i].begin(),
        this->LSTM_neurons[i].end(),
        [&](LSTM &n) {
          n.fire();
        });
  }

  for (int i = 0; i < predictions.size(); i++) {
    predictions[i] = 0;
  }
  for (int j = 0; j <  layer+ 1 ; j++) {
    std::for_each(
        std::execution::par_unseq,
        this->indexes.begin(),
        this->indexes.end(),
        [&](int index) {
          for (int i = 0; i < prediction_weights[index][j].size(); i++) {
            predictions[index] += prediction_weights[index][j][i] * this->LSTM_neurons[j][i].value;
          }
        });
  }
  for(int index = 0; index < predictions.size(); index++) {
    predictions[index] += bias[index];
    predictions[index] = sigmoid(predictions[index]);
  }
}

void DeepLSTM::backward(std::vector<float> targets, int layer) {

  for (int i = 0; i < targets.size(); i++) {
    errors[i] = targets[i] - predictions[i];
  }
//  Update the prediction weights
  for (int j = 0; j <  layer+ 1 ; j++) {
    std::for_each(
        std::execution::par_unseq,
        this->indexes_lstm_cells.begin(),
        this->indexes_lstm_cells.end(),
        [&](int index) {
          float gradient = 0;
          for (int i = 0; i < predictions.size(); i++) {
            gradient += errors[i] * prediction_weights[i][j][index] * predictions[i] * (1 - predictions[i]);
          }
          LSTM_neurons[j][index].zero_grad();
          LSTM_neurons[j][index].accumulate_gradient(gradient);
        });
  }

//  First we compute error, then we call accumulate gradient on the LSTM units


}

void DeepLSTM::update_parameters(int layer) {

  for (int j = 0; j <  layer+ 1 ; j++) {
  std::for_each(
      std::execution::par_unseq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        LSTM_neurons[j][index].update_weights(step_size);
      });
   }
  for (int l = 0; l < layer + 1; l++) {
    std::for_each(
        std::execution::par_unseq,
        this->indexes_lstm_cells.begin(),
        this->indexes_lstm_cells.end(),
        [&](int index) {
          for (int i = 0; i < predictions.size(); i++) {
            prediction_weights[i][l][index] += LSTM_neurons[l][index].value * errors[i] * step_size *predictions[i]*(1-predictions[i]);
          }
        });
  }
  for(int i = 0; i < predictions.size(); i++){
    bias[i] += errors[i]*step_size*predictions[i]*(1-predictions[i]);
  }
}

std::vector<float> DeepLSTM::read_output_values() {

  return predictions;
}

DeepLSTM::~DeepLSTM() {};