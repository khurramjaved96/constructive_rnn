//
// Created by Khurram Javed on 2021-09-28.
//

#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/recurrent_network_multilayer.h"
#include "../../../include/nn/lstm.h"

RecurrentNetworkMultilayer::RecurrentNetworkMultilayer(float step_size,
                                   int seed,
                                   int no_of_input_features,
                                   int total_targets,
                                   int total_recurrent_features_per_layer, int total_layers,
                                   int connections_per_feature) {

  this->mt.seed(seed);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> recurrent_weight_sampler(0, 0.99);
  for (int i = 0; i < no_of_input_features; i++) {
    Neuron *n = new LinearNeuron(true, false);
    n->neuron_age = 0;
    n->drinking_age = 20000;
    n->set_layer_number(0);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  float step_size_val = step_size;
  for (int output_neurons = 0; output_neurons < total_targets; output_neurons++) {
//    Neuron *output_neuron = new SigmoidNeuron(false, true);
    Neuron *output_neuron = new LinearNeuron(false, true);
    output_neuron->set_layer_number(100);
    this->all_neurons.push_back(output_neuron);
    this->output_neurons.push_back(output_neuron);
  }

  for(int layer_no = 0; layer_no< total_layers; layer_no++){
    int total_features_to_chose_from;
    if(layer_no == 0){
      total_features_to_chose_from = no_of_input_features;
    }
    else{
      total_features_to_chose_from = total_recurrent_features_per_layer;
    }
    std::uniform_int_distribution<int> neuron_number_sampler(0, total_features_to_chose_from - 1);
    std::vector<RecurrentRelu *> vector_of_features;
    for(int neuron_no = 0; neuron_no < total_recurrent_features_per_layer; neuron_no++){
      RecurrentRelu *recurent_neuron = new RecurrentRelu(false, false);
      recurent_neuron->set_layer_number(layer_no + 1 );
      vector_of_features.push_back(recurent_neuron);
      this->all_neurons.push_back(dynamic_cast<Neuron *>(recurent_neuron));
      Neuron *recurrent_neuron_base_class = recurent_neuron;
      Synapse
          *recurrent_synapse = new Synapse(recurrent_neuron_base_class,
                                           recurrent_neuron_base_class,
                                           recurrent_weight_sampler(this->mt),
                                           step_size_val);
      this->all_synapses.push_back(recurrent_synapse);
      std::vector<int> my_vec(total_features_to_chose_from, 0);
      for (int j = 0; j < connections_per_feature; j++) {
        int feature_index = -1;
        while (feature_index == -1) {
          int temp_index = neuron_number_sampler(this->mt);
          if (my_vec[temp_index] == 0) {
            my_vec[temp_index] = 1;
            feature_index = temp_index;
          }
        }
        Synapse *new_synapse;
        if(layer_no == 0)
          new_synapse = new Synapse(input_neurons[feature_index],
                                           recurrent_neuron_base_class,
                                           weight_sampler(this->mt),
                                           step_size_val);
        else{
          new_synapse = new Synapse(Recurrent_neuron_layer[layer_no-1][feature_index],
                                             recurrent_neuron_base_class,
                                             weight_sampler(this->mt),
                                             step_size_val);
        }
        this->all_synapses.push_back(new_synapse);
      }
    }
    this->Recurrent_neuron_layer.push_back(vector_of_features);

  }

  for (auto output_neuron: this->output_neurons) {
    for(auto vector_of_features: this->Recurrent_neuron_layer){
      for(int i = 0; i< vector_of_features.size(); i++){
        Synapse *output_synapse = new Synapse(vector_of_features[i], output_neuron, 0, step_size_val);
        this->output_synapses.push_back(output_synapse);
        this->all_synapses.push_back(output_synapse);
      }
    }
  }
}

RecurrentNetworkMultilayer::~RecurrentNetworkMultilayer() {

}

void RecurrentNetworkMultilayer::forward(std::vector<float> inp) {

  this->set_input_values(inp);

  std::for_each(
      std::execution::par_unseq,
      this->input_neurons.begin(),
      this->input_neurons.end(),
      [&](Neuron *n) {
        n->fire();
      });

  int counter = 0;
  for (auto RecurrentNeuronList: this->Recurrent_neuron_layer) {
    counter++;
//    std::cout << "Updating values " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        RecurrentNeuronList.begin(),
        RecurrentNeuronList.end(),
        [&](Neuron *n) {
          n->update_value();
        });

//    std::cout << "Firing " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        RecurrentNeuronList.begin(),
        RecurrentNeuronList.end(),
        [&](Neuron *n) {
          n->fire();
        });

//    std::for_each(
//        std::execution::par_unseq,
//        RecurrentNeuronList.begin(),
//        RecurrentNeuronList.end(),
//        [&](Neuron *n) {
//          n->update_utility();
//        });

  }


//  std::cout << "Updating values output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](Neuron *n) {
        n->update_value();
      });

//  std::cout << "Firing output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](Neuron *n) {
        n->fire();
      });

}

int RecurrentNetworkMultilayer::least_useful_feature() {
//  std::cout << "Getting lease useful feature index\n";
  float least_utility = 5000000;
  int least_useful_index = -1;
  int counter = 0;
  int total_eligible_for_replacement = 0;
  for (RecurrentRelu *neuron: this->Recurrent_neuron_layer[0]) {
//
    if (neuron->is_mature()) {
//      std::cout << "Found eligible neuron\n";
      total_eligible_for_replacement++;
    }
  }
  for (RecurrentRelu *neuron: this->Recurrent_neuron_layer[0]) {
    if(total_eligible_for_replacement*2 < this->Recurrent_neuron_layer[0].size()) {
//      std::cout << "Not enough old features\t" << total_eligible_for_replacement << " Tootal neurons "<< this->Recurrent_neuron_layer[0].size() <<  std::endl;
      return -1;
    }

    if (neuron->is_mature() && neuron->get_utility() < least_utility) {
      least_utility = neuron->get_utility();
      least_useful_index = counter;
    }
    counter++;
  }
  return least_useful_index;
}

void RecurrentNetworkMultilayer::replace_least_important_feature() {
  int index = least_useful_feature();
  std::cout << "Replacing feature = " << index << std::endl;
  if (index == -1)
    return;
  replace_feature(index);
}

void RecurrentNetworkMultilayer::replace_feature(int feature_no) {
//  std::cout << "Replacing feature no " << feature_no << "\n";
  std::uniform_int_distribution<int> neuron_number_sampler(0, this->input_neurons.size() - 1);
  std::uniform_real_distribution<float> weight_sampler(-1, 1);
  std::uniform_real_distribution<float> recurrent_weight_sampler(0, 0.95);
//  std::cout << "Getting recurrent feature\n";
  RecurrentRelu *feature = this->Recurrent_neuron_layer[0][feature_no];
  std::vector<int> my_vec(this->input_neurons.size(), 0);
//  std::cout << "Iterating over inputs\n";
//  std::cout << "Incoming synapses size " << feature->incoming_synapses.size() << std::endl;
  int total_incoming_synapses = feature->incoming_synapses.size();
  for (auto synapse: feature->incoming_synapses) {
//    std::cout << "In the for loop\n";
    int feature_index = -1;
    while (feature_index == -1) {
//      std::cout << "In the while loop\n";
      int temp_index = neuron_number_sampler(this->mt);
      if (my_vec[temp_index] == 0) {
        my_vec[temp_index] = 1;
        feature_index = temp_index;
      }
    }
//    std::cout << "Index found " << feature_index << "\n";
    synapse->input_neuron = this->input_neurons[feature_index];
    synapse->age = 0;
    synapse->weight = weight_sampler(this->mt);
  }
  feature->recurrent_synapse->weight = recurrent_weight_sampler(this->mt);
  feature->outgoing_synapses[0]->weight = 0;
  feature->outgoing_synapses[0]->age = 0;
  feature->neuron_age = 0;
//  std::cout << "Feature replaced\n";
}

void RecurrentNetworkMultilayer::reset_state() {
  std::for_each(
      std::execution::par_unseq,
      this->Recurrent_neuron_layer[0].begin(),
      this->Recurrent_neuron_layer[0].end(),
      [&](RecurrentRelu *n) {
        n->old_value= 0;
      });
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](Synapse *s) {
        s->TH = 0;
      });
}

void RecurrentNetworkMultilayer::backward(std::vector<float> target, int layer) {
  std::vector<float> error_list;
  int counter = 0;
  for(auto neuron: this->output_neurons){
    error_list.push_back(neuron->value - target[counter]);
    counter++;
  }

  std::for_each(
      std::execution::par_unseq,
      this->Recurrent_neuron_layer[layer].begin(),
      this->Recurrent_neuron_layer[layer].end(),
      [&](RecurrentRelu *n) {
        n->compute_gradient_of_all_synapses(error_list);
      });

  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](Synapse *s) {
        s->assign_credit();
      });

}

void RecurrentNetworkMultilayer::update_parameters() {

  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](Synapse *s) {
        s->update_weight();
      });
}



