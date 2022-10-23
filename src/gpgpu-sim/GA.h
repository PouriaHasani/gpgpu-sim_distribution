#ifndef GA_H
#define GA_H

#include "gpu-sim.h"
#include "power_stat.h"
#include "shader.h"
#include "vector"
#include "gpgpu_sim_wrapper.h"


class Population{
 public:
  Population(unsigned ,unsigned,unsigned ,double *,float,float,float);
  class Chromosome** parent_chromosomes;
  class Chromosome** child_chromosomes;
  double *available_alleles;
  unsigned number_avai_alleles;
  unsigned population_size;
  unsigned chromosome_size;
  double sum_powers_cons_one;
  double sum_powers_cons_zero;
  double sum_powers_time;
  float H_crossover;
  float H_fittest;
  float H_mutation;
  double *base_cluster_freq;
//Mcpat data to be past to objective function
  const gpgpu_sim_config *config;
  const shader_core_config *shdr_config;
  class gpgpu_sim_wrapper *wrapper;
  class power_stat_t *power_stats;
  unsigned stat_sample_freq;
  unsigned tot_cycle;
  unsigned cycle;
  unsigned tot_inst;
  unsigned inst;
  double base_freq;
  class simt_core_cluster **m_cluster;
  int shaders_per_cluster;
  float* numb_active_sms;
  double * cluster_freq;
  float* num_idle_core_per_cluster;
  float *average_pipeline_duty_cycle_per_sm;
  double Power;
  std::vector<double> Throughput;
  double* Max_Throughput;
  double Max_Throughput_scalar;
  double Total_Throughput;
//

  void population_init();
  unsigned parent_selection_SUS();
  void calculate_sum_of_powers();
  void calculate_cumulative_probability();
  void swap_mutation(unsigned parent_index,unsigned child_idx);
  void bit_flip_mutation(unsigned,unsigned );
  void one_point_crossover(unsigned first_parent,unsigned second_parent,unsigned child_idx);
  void All_chromosomes_Power();
  void Per_chromosomes_power(unsigned idx,unsigned );
  void mcpat_data_set(class gpgpu_sim_wrapper *wrapper,double Power,double* base_cluster_freq);
  class Binary_tree* power_tree;
  double* evolution(unsigned number_iterations);
  unsigned generate_random_values(unsigned min,unsigned max);
};

class Chromosome{
 public:
  Chromosome(unsigned);
  double *gene;
  unsigned size;
  double power;
  double Throughput;
  double power_time;
  int constraint;
  double power_calculation();
  float probability;
  float cumulative_probability;
  unsigned generate_random_values(unsigned min,unsigned max);
  void calculate_probability(double sum_power);

};

struct Node{
  double value;
  unsigned idx;
  int constraint;
  struct Node* left;
  struct Node* right;
};

struct array_data{
  unsigned idx;
  unsigned value;
  int constraint;
};

class Binary_tree{
 public:
  struct Node* root;
  struct Node* create_node(double value,unsigned idx);
  void create_tree(unsigned num_nodes,class Chromosome **);
  void insert(struct Node* root,double value,unsigned idx);
  void create_arrays(struct Node* node,std::vector<struct array_data>&,
                     std::vector<struct array_data>&);

  void print_(struct Node* root);

  void find_min(struct Node* root,unsigned &idx);
  void delete_min_node();
  unsigned counter;
  unsigned num_nodes;

};

#endif //GA_H
