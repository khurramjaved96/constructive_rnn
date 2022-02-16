//
// Created by Khurram Javed on 2022-01-17.
//


#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/td_lambda.h"
#include "../../../include/nn/neuron.h"
//
TDLambda::TDLambda(float step_size,
                   int seed,
                   int no_of_input_features,
                   int total_targets,
                   int total_recurrent_features,
                   int layer_size) {
  this->layer_size = layer_size;
  this->step_size = step_size;
  this->mt.seed(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(0, no_of_input_features + total_recurrent_features - 1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron n(true, false);
    this->input_neurons.push_back(n);
  }

  for (int i = 0; i < total_recurrent_features; i++) {
//    std::cout << "Recurrent feature no "<< i << std::endl;
    LSTM lstm_neuron(weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt),
                     weight_sampler(mt));
    for (int counter = 0; counter < this->input_neurons.size(); counter++) {
      Neuron *neuron_ref = &this->input_neurons[counter];
      lstm_neuron.add_synapse(neuron_ref,
                              weight_sampler(mt),
                              weight_sampler(mt),
                              weight_sampler(mt),
                              weight_sampler(mt));
    }
    indexes_lstm_cells.push_back(i);
    this->LSTM_neurons.push_back(lstm_neuron);
  }

  for (int counter = 0; counter < total_recurrent_features; counter++) {
    int layer_no = counter / layer_size;
    int incoming_features = 0;
    std::vector<int> map_index(no_of_input_features + total_recurrent_features, 0);
    while (incoming_features < no_of_input_features) {
      int index = index_sampler(mt);
      if (map_index[index] == 0) {
        map_index[index] = 1;
        if (index < no_of_input_features) {
//          std::cout << "Inp " << index << "\t" << counter << std::endl;
          incoming_features++;
          Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter].add_synapse(neuron_ref,
                                            weight_sampler(mt),
                                            weight_sampler(mt),
                                            weight_sampler(mt),
                                            weight_sampler(mt));
        } else {
          index = index - no_of_input_features;
          int new_layer_no = index / layer_size;
          if (new_layer_no < layer_no) {
//            std::cout << index << "\t" << counter << std::endl;
            incoming_features++;
            Neuron *neuron_ref = &this->LSTM_neurons[index];
            LSTM_neurons[counter].add_synapse(neuron_ref,
                                              weight_sampler(mt),
                                              weight_sampler(mt),
                                              weight_sampler(mt),
                                              weight_sampler(mt));
          }
        }
      }
    }
  }

  predictions = 0;
  bias = 0;
  bias_gradients = 0;
  for (int j = 0; j < this->LSTM_neurons.size(); j++) {
    prediction_weights.push_back(0);
    prediction_weights_gradient.push_back(0);

  }
}


float TDLambda::forward(std::vector<float> inputs) {
//  Set input neurons value
//  if(this->time_step%100000 == 99999)
//    this->step_size *= 0.8;
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = inputs[i];
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].update_value_sync();
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].compute_gradient_of_all_synapses();
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].fire();
  }

  predictions = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    predictions += prediction_weights[i] * this->LSTM_neurons[i].value;
  }
  predictions += bias;
  return predictions;
}


float TDLambda::get_target_without_sideeffects(std::vector<float> inputs) {
//  Backup old values
  std::vector<float> backup_vales;
  for (int i = 0; i < inputs.size(); i++) {
    backup_vales.push_back(this->input_neurons[i].value);
    this->input_neurons[i].value = inputs[i];
  }

//  Get hidden state without side-effects
  std::vector<float> hidden_state;
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    hidden_state.push_back(LSTM_neurons[i].get_value_without_sideeffects());
  }

//  Compute prediction
  float temp_prediction = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    temp_prediction += prediction_weights[i] * hidden_state[i];
  }
  temp_prediction += bias;


//  Restore values
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = backup_vales[i];
  }
  return temp_prediction;
}

void TDLambda::reset_state() {
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    LSTM_neurons[i].reset_state();
  }
}


void TDLambda::zero_grad() {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].zero_grad();
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] = 0;
  }

  bias_gradients = 0;

}

void TDLambda::decay_gradient(float decay_rate) {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter].decay_gradient(decay_rate);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] *= decay_rate;
  }

  bias_gradients *= decay_rate;

}

void TDLambda::backward() {

//  Update the prediction weights
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    float gradient = prediction_weights[index];
    LSTM_neurons[index].accumulate_gradient(gradient);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] += LSTM_neurons[index].value;
  }

  bias_gradients += 1;

}

void TDLambda::update_parameters(int layer, float error) {

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if (index <= (layer + 1) * layer_size)
      LSTM_neurons[index].update_weights(step_size, error);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if (index < (layer + 1) * layer_size) {
      prediction_weights[index] += prediction_weights_gradient[index] * error * step_size;
    }
  }

  bias += error * step_size * bias_gradients;

}

float TDLambda::read_output_values() {

  return predictions;
}

TDLambda::~TDLambda() {};