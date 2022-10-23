#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "GA.h"
#include "power_interface.h"
#include "vector"

Population::Population(unsigned population_size,unsigned chromosome_size,unsigned number_avai_alleles,\
                       double* available_alleles,float H_crossover,float H_fittest,float H_mutation){
  this->population_size = population_size;
  this->chromosome_size = chromosome_size;
  this->available_alleles = available_alleles;
  this->number_avai_alleles = number_avai_alleles;
  this->H_crossover = H_crossover;
  this->H_fittest = H_fittest;
  this->H_mutation = H_mutation;

  parent_chromosomes = (class Chromosome**)malloc(sizeof(class Chromosome*)*population_size);
  child_chromosomes = (class Chromosome**)malloc(sizeof(class Chromosome*)*population_size);
  power_tree = (class Binary_tree*) malloc(sizeof(class Binary_tree));
  for(unsigned i=0;i<population_size;i++) {
    parent_chromosomes[i] = (class Chromosome*)malloc(sizeof(class Chromosome)*population_size);
    child_chromosomes[i] = (class Chromosome*)malloc(sizeof(class Chromosome)*population_size);
    *(parent_chromosomes[i]) = Chromosome(chromosome_size);
    *(child_chromosomes[i]) = Chromosome(chromosome_size);
  }
}

void Population::mcpat_data_set(class gpgpu_sim_wrapper *wrapper,double Power,double* base_cluster_freq){

  this->wrapper = wrapper;
  this->base_cluster_freq = base_cluster_freq;
  this->Power = Power+59;
}

void Chromosome::calculate_probability(double power_sum_time){
  probability = power_time/power_sum_time;
}

void Population::calculate_sum_of_powers(){
  sum_powers_time = 0;
  for(unsigned i=0;i<population_size;i++)
    sum_powers_time += this->parent_chromosomes[i]->power_time;
}

void Population:: calculate_cumulative_probability(){
  float sum = 0;
  unsigned idx;
  for(unsigned i=0;i<population_size;i++){
    parent_chromosomes[i]->calculate_probability(sum_powers_time);
    sum+=parent_chromosomes[i]->probability;
    parent_chromosomes[i]->cumulative_probability = sum;
  }
}
//Select parents with better fitness value with higher probability
unsigned Population::parent_selection_SUS(){
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(0, RAND_MAX); // define the range
  float r = (float)(distr(gen))/RAND_MAX;
  unsigned index=0;

  while(index<population_size-1 && parent_chromosomes[index]->cumulative_probability<r){
    index++;
  }

  return index;
}

void Population::swap_mutation(unsigned parent_index,unsigned child_idx){

  for(int i=0;i<child_chromosomes[0]->size;i++)
    child_chromosomes[child_idx]->gene[i] = parent_chromosomes[parent_index]->gene[i];

  unsigned first_gene = generate_random_values(0, parent_chromosomes[parent_index]->size-1);
  unsigned second_gene = generate_random_values(0, parent_chromosomes[parent_index]->size-1);;
  child_chromosomes[child_idx]->gene[first_gene] = parent_chromosomes[parent_index]->gene[second_gene];
  child_chromosomes[child_idx]->gene[second_gene] = parent_chromosomes[parent_index]->gene[first_gene];
}

void Population::bit_flip_mutation(unsigned parent_index,unsigned child_idx){
  for(int i=0;i<child_chromosomes[0]->size;i++)
    child_chromosomes[child_idx]->gene[i] = parent_chromosomes[parent_index]->gene[i];
  unsigned gene_idx = generate_random_values(0, parent_chromosomes[0]->size-1);
  unsigned freq_idx = generate_random_values(0,number_avai_alleles-1);
  child_chromosomes[child_idx]->gene[gene_idx] = available_alleles[freq_idx];
}

void Population::one_point_crossover(unsigned first_parent,unsigned second_parent,unsigned child_idx) {
  unsigned point = generate_random_values(0, child_chromosomes[child_idx]->size - 1);
  for (unsigned i = 0; i < child_chromosomes[child_idx]->size; i++) {
    if (i <= point) {
      child_chromosomes[child_idx]->gene[i] = parent_chromosomes[first_parent]->gene[i];
      child_chromosomes[child_idx + 1]->gene[i] = parent_chromosomes[second_parent]->gene[i];
    } else {
      child_chromosomes[child_idx]->gene[i] = parent_chromosomes[second_parent]->gene[i];
      child_chromosomes[child_idx + 1]->gene[i] = parent_chromosomes[first_parent]->gene[i];
    }
  }
}

unsigned Population::generate_random_values(unsigned min, unsigned max) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(min, max); // define the range
  return (unsigned)(distr(gen));
}

void Population::All_chromosomes_Power() {
  for(int i=1;i<population_size;i++){
    mcpat_cycle_power_calculation(parent_chromosomes[i],
                                  wrapper,
                                  base_cluster_freq);
  }
}
//Calculate fitness values
void Population::Per_chromosomes_power(unsigned idx,unsigned p){
  if(p) {
    mcpat_cycle_power_calculation(
            parent_chromosomes[idx],
            wrapper ,base_cluster_freq);
  }
  else {
    mcpat_cycle_power_calculation(child_chromosomes[idx],
        wrapper ,base_cluster_freq);
  }
}
//Initialize the first population
void Population::population_init(){
  unsigned idx;
 double power = 0;
 unsigned I = 0;
  for (unsigned i = 0; i < parent_chromosomes[0]->size; i++) {
      parent_chromosomes[0]->gene[i] = base_cluster_freq[i];
      power+=wrapper->sample_cmp_pwr_S[i]*base_cluster_freq[i]/(700*1e6)+wrapper->sample_cmp_pwr_Shrd/15 + wrapper->sample_cmp_pwr_const /15;
      I += wrapper->I[i];
  }

  parent_chromosomes[0]->power = power+59;
  parent_chromosomes[0]->power_time = power/I;

  for (unsigned i = 1; i < population_size; i++) {
    for (unsigned j = 0; j < parent_chromosomes[i]->size; j++) {
      idx = generate_random_values(0, number_avai_alleles - 1);
      parent_chromosomes[i]->gene[j] = available_alleles[idx];
    }
  }

  //Calculate fitness value for all the chromosomes
  All_chromosomes_Power();
}

double* Population::evolution(unsigned number_iterations) {
    //    create first generation with random values
    population_init();
    unsigned idx;
    unsigned idy;
    unsigned child_chromosome_idx;
    double temp;
    unsigned muted_chromosome;
    unsigned second_parent;
    unsigned first_parent;
    class Chromosome **temp_chromosome;

    for (unsigned itr = 0; itr < number_iterations; itr++) {
        child_chromosome_idx = 0;

        //    create binary tree for finding min in the power values
        power_tree->create_tree(population_size, parent_chromosomes);
        calculate_sum_of_powers();
        calculate_cumulative_probability();

        //    find top 10% best fits
        for (int k = 0; k < (int) (population_size * H_fittest); k++) {
            power_tree->find_min(power_tree->root,idx);
            power_tree->print_(power_tree->root);
            power_tree->delete_min_node();
            child_chromosome_idx++;
            for (int i = 0; i < child_chromosomes[idx]->size; i++) {
                child_chromosomes[k]->gene[i] = parent_chromosomes[idx]->gene[i];
            }
            child_chromosomes[k]->power = parent_chromosomes[idx]->power;
            child_chromosomes[k]->power_time = parent_chromosomes[idx]->power_time;
        }

        ////    swap mutation
        
        for (int k = 0; k < (int) (population_size * H_mutation / 2); k++) {
            muted_chromosome = parent_selection_SUS();
            swap_mutation(muted_chromosome, child_chromosome_idx);
            Per_chromosomes_power(child_chromosome_idx, 0);

            child_chromosome_idx++;
        }

        for (int k = 0; k < (int) (population_size * H_mutation / 2); k++) {
            muted_chromosome = parent_selection_SUS();
            bit_flip_mutation(muted_chromosome, child_chromosome_idx);
            Per_chromosomes_power(child_chromosome_idx, 0);
            child_chromosome_idx++;
        }
        //    CrossOver
        for (int k = 0; k < (int) (population_size * H_crossover); k++) {

            first_parent = parent_selection_SUS();
            second_parent = parent_selection_SUS();

            one_point_crossover(first_parent, second_parent, child_chromosome_idx);
            Per_chromosomes_power(child_chromosome_idx, 0);
            Per_chromosomes_power(child_chromosome_idx + 1, 0);
            child_chromosome_idx += 2;
        }

        temp_chromosome = child_chromosomes;
        child_chromosomes = parent_chromosomes;
        parent_chromosomes = temp_chromosome;

}
  return child_chromosomes[0]->gene;

}

Chromosome::Chromosome(unsigned size){
  this->size = size;
  gene = (double*)malloc(sizeof(double)*size);
  if(gene == NULL)
  {
    std::cout<<"Couldn't allocate memory for new chromozome"<<std::endl;
    exit(0);
  }
};

unsigned Chromosome::generate_random_values(unsigned min, unsigned max) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(min, max); // define the range
  return (unsigned)(distr(gen));
}


double Chromosome::power_calculation(){
  power = 0;
  for(int j=0;j<size;j++)
    power += pow(gene[j],2);
  return power;
}

struct Node* Binary_tree::create_node(double value,unsigned idx) {
  struct Node* node;
  node = (struct Node*)malloc(sizeof(struct Node));
  node->value = value;
  node->idx = idx;
  node->right = NULL;
  node->left = NULL;
  return node;
}

void Binary_tree::insert(struct Node* root,double value,unsigned idx){
  if(root == NULL)
    root = create_node(value,idx);
  else{
    if(value <= root->value ){
      if(root->left == NULL)
        root->left = create_node(value,idx);
      else
        insert(root->left,value,idx);
    }
    else
      if(root->right == NULL)
         root->right = create_node(value,idx);
      else
        insert(root->right,value,idx);
  }
}



void Binary_tree::create_tree(unsigned int num_nodes, class Chromosome **chromosome_tree) {
  this->num_nodes = num_nodes;
  root = create_node(chromosome_tree[0]->power_time,0);
  this->num_nodes = num_nodes;
  for(unsigned i = 1;i<num_nodes;i++) {
    insert(root, chromosome_tree[i]->power_time, i);
    std::cout<<chromosome_tree[i]->power_time<<"\tpower_time: "<<root->value<<std::endl;
  }
}

void Binary_tree::create_arrays(struct Node* node,std::vector<struct array_data>&Array_con_one,std::vector<struct array_data>&Array_con_zero){
  if(node == NULL)
    return ;
  create_arrays(node->left,Array_con_one,Array_con_zero);

  struct array_data temp;
  temp.constraint = node->constraint;
  temp.value = node->value;
  temp.idx = node->idx;

  if(node->constraint == 1) {
    Array_con_one.push_back(temp);
  }
  else
    Array_con_zero.push_back(temp);
  create_arrays(node->right,Array_con_one,Array_con_zero);
}

void Binary_tree::print_(struct Node* root) {
    if(root == NULL)
        return;
        print_(root->left);
    std::cout<<"\ttree: "<<root->value<<std::endl;
    print_(root->right);

}

void Binary_tree::find_min(struct Node* root,unsigned &idx){
    if(root->left == NULL)
        idx = root->idx;
    else
        find_min(root->left,idx);
}

void Binary_tree::delete_min_node(){
  struct Node* parent_node = root;
  if(root->left == NULL){
    root = root->right;
    return;
  }
  struct Node* left_node = root->left;
  while(left_node->left != NULL){
    parent_node = left_node;
    left_node  = left_node->left;
  }
  parent_node->left = left_node->right;
};