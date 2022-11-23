#include <iostream>
#include <math.h>
#include "include/nn/networks/lstm_bptt.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"

#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"

int main(int argc, char *argv[]) {
  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector<std::string>{"run", "step", "error"},
                               std::vector<std::string>{"int", "int", "real"},
                               std::vector<std::string>{"run", "step"});

  Metric avg_error = Metric(my_experiment.database_name, "predictions",
                            std::vector<std::string>{"run", "global_step", "step", "pred", "target"},
                            std::vector<std::string>{"int", "int", "int", "real", "real"},
                            std::vector<std::string>{"run", "global_step"});
  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment.get_int_param("seed"));

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");

  int total_data_points = 60000;
  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);

  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;

  auto network = DenseLSTM(my_experiment.get_float_param("step_size"),
                           my_experiment.get_int_param("seed"),
                           my_experiment.get_int_param("features"),
                           28,
                           my_experiment.get_int_param("truncation"));

  for (int counter = 0; counter < total_data_points; counter++) {
    std::vector<float> x_temp;
    for (auto inner: dataset.training_images[counter]) {
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    if (int(unsigned(dataset.training_labels[counter])) % 2 == 0)
      y_temp.push_back(1.0);
    else
      y_temp.push_back(0.0);
    y_temp.push_back(float(unsigned(dataset.training_labels[counter])));
    images.push_back(x_temp);
    targets.push_back(y_temp);
  }

  int ITI_steps = my_experiment.get_int_param("ITI");
  float running_error = 0.05;
  float gamma = 0.8; // err @ 0 pred: 0.2 @ 20 step, 0.04 @ 14 step, 0.02 @ 0 step
  float accuracy = 0.1;

  std::cout << images.size() << " " << targets.size() << std::endl;
  int global_step = 0;
  std::vector<float> row_x;
  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {
    int index = index_sampler(mt);
//    std::cout << "INdex =  " << index << std::endl;
    auto x = images[index];
    float y = targets[index][0];
    // the last step will be ignored but it doesnt matter
    // just make sure ITI_steps > 0
    row_x.clear();
    for (int temp = 0 * 28; temp < (0 + 1) * 28; temp++)
      row_x.push_back(x[temp] / 256.0);

    for (int row = 1; row < 28 + ITI_steps; row++) {
      global_step += 1;

      float pred = network.forward(row_x);
      float target = 0;
      float return_target = 0;

      row_x.clear();
      for (int temp = row * 28; temp < (row + 1) * 28; temp++) {
        if (row < 28)
          row_x.push_back(x[temp] / 256.0);
        else
          row_x.push_back(0.0);
      }

      if (row < 28)
        return_target = y * pow(gamma, 27 - row);
      if (row == 27)
        target = y + gamma * network.get_target_without_sideeffects(row_x);
      else
        target = 0 + gamma * network.get_target_without_sideeffects(row_x);
      float error = target - pred;
      float return_error = (return_target - pred) * (return_target - pred);
      running_error = running_error * 0.9999 + 0.0001 * return_error;
      network.decay_gradient(my_experiment.get_float_param("lambda") * gamma);
      network.backward();
      network.update_parameters(error);
      if (i % 10000 < 20) {
        std::vector<std::string> cur_error;
        cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
        cur_error.push_back(std::to_string(global_step));
        cur_error.push_back(std::to_string(i));
        cur_error.push_back(std::to_string(pred));
        cur_error.push_back(std::to_string(return_target));
        print_vector(cur_error);
        avg_error.record_value(cur_error);
      }
    }

    if (i % 5000 == 2000) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error));
      error_metric.record_value(cur_error);
    }

    if (i % 1000 == 0) {
      std::cout << "Step = " << i << std::endl;
      std::cout << "Error= " << running_error << std::endl;
    }

    if (i % 15000 == 0) {
      error_metric.commit_values();
      avg_error.commit_values();
    }
  }
}
