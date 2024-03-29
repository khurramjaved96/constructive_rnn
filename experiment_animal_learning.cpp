//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/nn/networks/lstm_incremental_networks.h"
#include "include/nn/networks/lstm_bptt.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"
#include <string>
#include "include/nn/networks/network_factory.h"
#include "include/nn/networks/base_lstm.h"
#include "include/environments/animal_learning/tracecondioning.h"

int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "error"},
                               std::vector<std::string>{"int", "int", "real"},
                               std::vector<std::string>{"run", "step"});

  Metric avg_error = Metric(my_experiment->database_name, "predictions",
                            std::vector<std::string>{"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "pred",
                                                     "target"},
                            std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "real",
                                                     "real", "real", "real"},
                            std::vector<std::string>{"run", "step"});
//  Metric network->state = Metric(my_experiment->database_name, "network->state",
//                                std::vector<std::string>{"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
//                                                         "x8", "x9"},
//                                std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "real",
//                                                         "real", "real", "real", "real"},
//                                std::vector<std::string>{"run", "step"});

  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment->get_int_param("seed"));
  std::pair<int, int> ISI{14, 26};
  std::pair<int, int> ITI{80, 120};
  TracePatterning env = TracePatterning(ISI, ITI, 5, my_experiment->get_int_param("seed"));

  BaseLSTM *network = NetworkFactory::get_network(my_experiment);

  std::cout << "Network created\n";
  float running_error = 0.05;
  auto x = env.reset();
  int layer = 0;
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {

    if (i % my_experiment->get_int_param("freq") == my_experiment->get_int_param("freq") - 1) {
      layer++;
      std::cout << "Increasing layer\n";
    }

    float gamma = 0.90;
    float pred = network->forward(x);
    float real_target = env.get_target(gamma);

    if (i % 1000000 < 400) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      for (int inner_c = 0; inner_c < 7; inner_c++) {
        cur_error.push_back(std::to_string(x[inner_c]));
      }
      cur_error.push_back(std::to_string(pred));
      cur_error.push_back(std::to_string(real_target));
      avg_error.record_value(cur_error);
    }

    x = env.step();
    float target = env.get_US() + gamma * network->get_target_without_sideeffects(x);

    float error = target - pred;
    float real_error = (real_target - pred) * (real_target - pred);
    running_error = running_error * 0.99999 + 0.00001 * real_error;
    network->decay_gradient(my_experiment->get_float_param("lambda") * gamma);
    network->backward(layer);
    network->update_parameters(layer, error);
    if (i % 50000 == 20000) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error));
      error_metric.record_value(cur_error);
    }
//
    if (i % 500000 == 0) {
      std::cout << "Step = " << i << " Error = " << running_error << std::endl;
      error_metric.commit_values();
      avg_error.commit_values();
//      network->state.commit_values();
    }
  }
  error_metric.commit_values();
  avg_error.commit_values();

}
