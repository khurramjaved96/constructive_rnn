#include <iostream>
#include <math.h>
#include <chrono>
#include "include/nn/networks/lstm_incremental_networks.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"

#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"


int main(int argc, char *argv[]) {
  auto start = std::chrono::steady_clock::now();
  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"run", "step", "error"},
                               std::vector < std::string > {"int", "int", "real"},
                               std::vector < std::string > {"run", "step"});

  Metric avg_error = Metric(my_experiment.database_name, "predictions",
                            std::vector < std::string > {"run", "global_step", "step", "pred", "target"},
                            std::vector < std::string > {"int", "int", "int", "real", "real"},
                            std::vector < std::string > {"run", "global_step"});
  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment.get_int_param("seed"));

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");

  int total_data_points = 60000;
  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);


  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;


  auto network = TDLambda(my_experiment.get_float_param("step_size"),
                       my_experiment.get_int_param("seed"),
                       1,
                       1,
                       my_experiment.get_int_param("features"),
                       my_experiment.get_int_param("width"));

  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
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
  float gamma = 0.99;
  float accuracy = 0.1;

  std::cout << images.size() << " " << targets.size() << std::endl;
  int layer = 0;
  int global_step = 0;
  std::vector<float> pixel_x;
  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {
    if(i % my_experiment.get_int_param("freq") == my_experiment.get_int_param("freq") - 1){
      layer++;
      std::cout << "Increasing layer\n";
    }

    int index = index_sampler(mt);
//    std::cout << "INdex =  " << index << std::endl;
    auto x = images[index];
    float y = targets[index][0];
    // Last ITI step will be ignored but it doesnt matter
    pixel_x.clear();
    pixel_x.push_back(x[0]/256.0);

    for(int pixel_idx= 1; pixel_idx < 784 + ITI_steps; pixel_idx++){
      global_step += 1;

      float pred = network.forward(pixel_x);
      float target = 0;
      float return_target = 0;
      std::vector<float> pixel_x;
      pixel_x.clear();
      if (pixel_idx < 784)
        pixel_x.push_back(x[pixel_idx]/256.0);
      else
        pixel_x.push_back(0.0);
      if (pixel_idx < 784)
        return_target = y*pow(gamma, 783 - pixel_idx);
      if (pixel_idx == 783)
        target = y + gamma * network.get_target_without_sideeffects(pixel_x);
      else
        target = 0.0 + gamma * network.get_target_without_sideeffects(pixel_x);
      float error = target - pred;
      float return_error = (return_target - pred) * (return_target - pred);
      running_error = running_error*0.9999 + 0.0001*return_error;
      network.decay_gradient(my_experiment.get_float_param("lambda")*gamma);
      network.backward();
      network.update_parameters(layer, error);
      if(i%10000 < 20){
        std::vector<std::string> cur_error;
        cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
        cur_error.push_back(std::to_string(global_step));
        cur_error.push_back(std::to_string(i));
        cur_error.push_back(std::to_string(pred));
        cur_error.push_back(std::to_string(return_target));
        //std::cout << pixel_x[0] << " " << y << std::endl;
        //std::cout << target << std::endl;
        print_vector(cur_error);
        avg_error.record_value(cur_error);
      }
    }

    if(i%5000 == 2000){
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment.get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error));
      error_metric.record_value(cur_error);
    }

    if(i % 1000 == 0){
      std::cout << "Step = " << i << std::endl;
      std::cout << "Error= " << running_error << std::endl;
    }

    if(i%15000 == 0){
      error_metric.commit_values();
      avg_error.commit_values();
    }

    if(i % 1000 == 0){
      std::cout << "Step = " << i << std::endl;
      std::cout << "Error= " << running_error << std::endl;
      auto end = std::chrono::steady_clock::now();
      std::cout << "Elapsed time in milliseconds for per steps: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " ms" << std::endl;
      std::cout << "Elapsed time in milliseconds for per steps: "
                << std::chrono::duration_cast<std::chrono::duration<double>>(end- start).count()
                << " seconds" << std::endl;
      std::cout << "Elapsed time in milliseconds for per steps: "
                << 1000000 / (1+(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                              my_experiment.get_int_param("steps")))
                << " fps" << std::endl;
      start = std::chrono::steady_clock::now();

    }
  }
}
