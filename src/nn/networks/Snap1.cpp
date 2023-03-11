//
// Created by Khurram Javed on 2022-01-17.
//


#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/snap1.h"
#include "../../../include/nn/neuron.h"
//
Snap1::Snap1(float step_size,
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
                     weight_sampler(mt),
                     0.01, 0.99999);
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

  for(int counter = 0; counter < total_recurrent_features; counter++){
    int layer_no = counter/layer_size;
    int incoming_features = 0;
    std::vector<int> map_index(no_of_input_features + total_recurrent_features, 0);
    while(incoming_features < no_of_input_features){
      int index = index_sampler(mt);
      if(map_index[index] == 0){
        map_index[index] = 1;
        if(index < no_of_input_features){
//          std::cout << "Inp " << index << "\t" << counter << std::endl;
          incoming_features++;
          Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter].add_synapse(neuron_ref,
                                  weight_sampler(mt),
                                  weight_sampler(mt),
                                  weight_sampler(mt),
                                  weight_sampler(mt));
        }
        else{
          index = index - no_of_input_features;
          int new_layer_no = index/ layer_size;
          if(new_layer_no < layer_no) {
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

//  std::cout << "From\tTo\n";
//  for (int outer = 0; outer < total_recurrent_features; outer++) {
//    for (int inner = 0; inner < total_recurrent_features; inner++) {
//      int temp_outer = outer/ layer_size;
//      int temp_inner = inner/ layer_size;
//
//      if (temp_outer < temp_inner) {
////        std::cout << outer << "\t" << inner <<std::endl;
//        Neuron *neuron_ref = &this->LSTM_neurons[outer];
//        LSTM_neurons[inner].add_synapse(neuron_ref,
//                                        weight_sampler(mt),
//                                        weight_sampler(mt),
//                                        weight_sampler(mt),
//                                        weight_sampler(mt));
//      }
//    }
//  }

  for (int i = 0; i < total_targets; i++) {
    predictions.push_back(0);
    errors.push_back(0);
    indexes.push_back(i);
    bias.push_back(0);
    std::vector<float> prediction_weight_cur_target;
    for (int j = 0; j < this->LSTM_neurons.size(); j++) {
      prediction_weight_cur_target.push_back(0);
    }
    prediction_weights.push_back(prediction_weight_cur_target);
  }
}

void Snap1::reset_state() {
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    LSTM_neurons[i].reset_state();
  }
}

void Snap1::forward(std::vector<float> inputs) {
//  Set input neurons value
//  if(this->time_step%100000 == 99999)
//    this->step_size *= 0.8;
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = inputs[i];
  }
  std::for_each(
      std::execution::seq,
      this->LSTM_neurons.begin(),
      this->LSTM_neurons.end(),
      [&](LSTM &n) {
        n.update_value_sync();

      });

  std::for_each(
      std::execution::seq,
      this->LSTM_neurons.begin(),
      this->LSTM_neurons.end(),
      [&](LSTM &n) {
        n.compute_gradient_of_all_synapses();
      });

  std::for_each(
      std::execution::seq,
      this->LSTM_neurons.begin(),
      this->LSTM_neurons.end(),
      [&](LSTM &n) {
        n.fire();
      });

  std::for_each(
      std::execution::seq,
      this->indexes.begin(),
      this->indexes.end(),
      [&](int index) {
        predictions[index] = 0;
        for (int i = 0; i < prediction_weights[index].size(); i++) {
          predictions[index] += prediction_weights[index][i] * this->LSTM_neurons[i].value;
        }
        predictions[index] += bias[index];
        predictions[index] = sigmoid(predictions[index]);
      });


//  std::cout << "Predictions = ";
//  for(int i = 0; i < predictions.size(); i++)
//    std::cout << predictions[i] << ",";
//  std::cout << std::endl;
}

float Snap1::backward(std::vector<float> targets) {

  for (int i = 0; i < targets.size(); i++) {
    errors[i] = targets[i] - predictions[i];
  }
//  Update the prediction weights
  std::for_each(
      std::execution::seq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        float gradient = 0;
        for (int i = 0; i < predictions.size(); i++) {
          gradient += errors[i] * prediction_weights[i][index] * predictions[i] * (1 - predictions[i]);
        }
        LSTM_neurons[index].zero_grad();
        LSTM_neurons[index].accumulate_gradient(gradient);
      });
  float error_sum = 0;
  for (auto er : errors) {
    error_sum += std::abs(er);
  }
  return error_sum;
//  First we compute error, then we call accumulate gradient on the LSTM units


}

void Snap1::update_parameters(int layer) {
  int neurons_in_layer = layer* layer_size;
//  int end_range = min(neurons_in_layer+ layer_size, LSTM_neurons.size());
//  for(int temp_counter = neurons_in_layer; temp_counter < end_range; temp_counter++)
//    LSTM_neurons[temp_counter].update_weights(step_size);
//  for (int i = 0; i < pred++) {
//    prediction_weights[i][layer] +=
//        LSTM_neurons[layer].value * errors[i] * step_size * predictions[i] * (1 - predictions[i]);
//  }


  std::for_each(
      std::execution::seq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        if(index <= (layer+1) *  layer_size)
          LSTM_neurons[index].update_weights(step_size);
      });

  std::for_each(
      std::execution::seq,
      this->indexes_lstm_cells.begin(),
      this->indexes_lstm_cells.end(),
      [&](int index) {
        if(index < (layer+1) *  layer_size) {
          for (int i = 0; i < predictions.size(); i++) {
            prediction_weights[i][index] +=
                LSTM_neurons[index].value * errors[i] * step_size * predictions[i] * (1 - predictions[i]);
          }
        }
      }
  );

  for (int i = 0; i < predictions.size(); i++) {
    bias[i] += errors[i] * step_size * predictions[i] * (1 - predictions[i]);
  }
//  std::cout << "Bias = " << bias[0] << std::endl;
//  std::cout << "Bias = " << bias[1] << std::endl;

}

std::vector<float> Snap1::read_output_values() {

  return predictions;
}

Snap1::~Snap1() {};