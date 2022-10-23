// Copyright (c) 2009-2011, Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "power_interface.h"
#include "GA.h"
#include <fstream>
void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst) {
  wrapper->init_mcpat(
      config.g_power_config_name, config.g_power_filename,
      config.g_power_trace_filename, config.g_metric_trace_filename,
      config.g_steady_state_tracking_filename,
      config.g_power_simulation_enabled, config.g_power_trace_enabled,
      config.g_steady_power_levels_enabled, config.g_power_per_cycle_dump,
      config.gpu_steady_power_deviation, config.gpu_steady_min_period,
      config.g_power_trace_zlevel, tot_inst + inst, stat_sample_freq);
}

bool mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst,class simt_core_cluster **m_cluster,int shaders_per_cluster,\
                 float* numb_active_sms,double * cluster_freq,float* average_pipeline_duty_cycle_per_sm
                    ,double &Total_exe_time,double* new_cluster_freq,double *pre_cluster_freq,double p_model_freq,int gpu_stall_dramfull,bool &first) {
  static bool mcpat_init = true;

  if (mcpat_init) {  // If first cycle, don't have any power numbers yet
    mcpat_init = false;
    return false;
  }
  //If sampling cycle reached 700 cycles the power calculation and frequency optimizations are done here
  if ((tot_cycle + cycle) % stat_sample_freq == 0) {

    Total_exe_time += (stat_sample_freq/p_model_freq);
    //Declare and allocate arrays to account for per Cluster data collection
    //This dynamic memory allocations for per cluster data collections are added for all the performance counter
    //captured with GPGPU-SIM
    double *tot_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *total_int_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *tot_fp_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *tot_commited_ins_set_inst_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);




    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, stat_sample_freq,
        stat_sample_freq, power_stats->get_total_inst(tot_ins_set_inst_power),
        power_stats->get_total_int_inst(total_int_ins_set_inst_power),
        power_stats->get_total_fp_inst(tot_fp_ins_set_inst_power),
        power_stats->get_l1d_read_accesses(),
        power_stats->get_l1d_write_accesses(),
        power_stats->get_committed_inst(tot_commited_ins_set_inst_power),
        tot_ins_set_inst_power, total_int_ins_set_inst_power,
        tot_fp_ins_set_inst_power, tot_commited_ins_set_inst_power,cluster_freq,stat_sample_freq);


    // Single RF for both int and fp ops
    double *regfile_reads_set_regfile_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *regfile_writes_set_regfile_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *non_regfile_operands_set_regfile_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);


    wrapper->set_regfile_power(
        power_stats->get_regfile_reads(regfile_reads_set_regfile_power),
        power_stats->get_regfile_writes(regfile_writes_set_regfile_power),
        power_stats->get_non_regfile_operands(
            non_regfile_operands_set_regfile_power),
        regfile_reads_set_regfile_power, regfile_writes_set_regfile_power,
        non_regfile_operands_set_regfile_power);

    // Instruction cache stats
    wrapper->set_icache_power(power_stats->get_inst_c_hits(),
                              power_stats->get_inst_c_misses());

    // Constant Cache, shared memory, texture cache
    wrapper->set_ccache_power(power_stats->get_constant_c_hits(),
                              power_stats->get_constant_c_misses());
    wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
                              power_stats->get_texture_c_misses());

    double *shmem_read_set_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    wrapper->set_shrd_mem_power(
        power_stats->get_shmem_read_access(shmem_read_set_power),
        shmem_read_set_power);

    wrapper->set_l1cache_power(
        power_stats->get_l1d_read_hits(), power_stats->get_l1d_read_misses(),
        power_stats->get_l1d_write_hits(), power_stats->get_l1d_write_misses());

    wrapper->set_l2cache_power(
        power_stats->get_l2_read_hits(), power_stats->get_l2_read_misses(),
        power_stats->get_l2_write_hits(), power_stats->get_l2_write_misses());

    float active_sms = 0;
    FILE *file;

    float *active_sms_per_cluster =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float *num_cores_per_cluster =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float *num_idle_core_per_cluster =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);

    float total_active_sms = 0;
    for (int i = 0; i < wrapper->number_shaders; i++) {
      active_sms_per_cluster[i] = numb_active_sms[i] * ((p_model_freq) /
                                  cluster_freq[i] )/ stat_sample_freq;
      active_sms += active_sms_per_cluster[i];
      num_cores_per_cluster[i] = shaders_per_cluster;
      num_idle_core_per_cluster[i] =
          num_cores_per_cluster[i] - active_sms_per_cluster[i];

    }

    float num_cores = shdr_config->num_shader();
    float num_idle_core = num_cores - active_sms;

    wrapper->set_idle_core_power(num_idle_core, num_idle_core_per_cluster);

    float* pipeline_duty_cycle_per_sm = (float*)malloc(sizeof(float) *wrapper->number_shaders);
    for (int i = 0; i < wrapper->number_shaders; i++) {
      pipeline_duty_cycle_per_sm[i] = average_pipeline_duty_cycle_per_sm[i] * (p_model_freq/cluster_freq[i]) / stat_sample_freq < 0.8\
          ?average_pipeline_duty_cycle_per_sm[i] * (p_model_freq/cluster_freq[i]) / stat_sample_freq : 0.8;

    }

    float pipeline_duty_cycle =
        ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) <
         0.8)
            ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq)
            : 0.8;
    wrapper->set_duty_cycle_power(pipeline_duty_cycle,pipeline_duty_cycle_per_sm);


    // Memory Controller
    wrapper->set_mem_ctrl_power(power_stats->get_dram_rd(),
                                power_stats->get_dram_wr(),
                                power_stats->get_dram_pre());

    // Execution pipeline accesses
    // FPU (SP) accesses, Integer ALU (not present in Tesla), Sfu accesses
    double *tot_fpu_accessess_set_exec_unit_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *ialu_accessess_set_exec_unit_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *tot_sfu_accessess_set_exec_unit_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);

    wrapper->set_exec_unit_power(
        power_stats->get_tot_fpu_accessess(
            tot_fpu_accessess_set_exec_unit_power),
        power_stats->get_ialu_accessess(ialu_accessess_set_exec_unit_power),
        power_stats->get_tot_sfu_accessess(
            tot_sfu_accessess_set_exec_unit_power),
        tot_fpu_accessess_set_exec_unit_power,
        ialu_accessess_set_exec_unit_power,
        tot_sfu_accessess_set_exec_unit_power);

    // Average active lanes for sp and sfu pipelines
    float *sp_active_lanes_set_active_lanes_power =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float *sfu_active_lanes_set_active_lanes_power =
        (float *)malloc(sizeof(float) * wrapper->number_shaders);
    float avg_sp_active_lanes = power_stats->get_sp_active_lanes(
        sp_active_lanes_set_active_lanes_power, cluster_freq,p_model_freq, stat_sample_freq);
    float avg_sfu_active_lanes = (power_stats->get_sfu_active_lanes(
        sfu_active_lanes_set_active_lanes_power, cluster_freq,p_model_freq,
        stat_sample_freq));
    assert(avg_sp_active_lanes <= 32);
    assert(avg_sfu_active_lanes <= 32);
    wrapper->set_active_lanes_power((power_stats->get_sp_active_lanes(
                                        sp_active_lanes_set_active_lanes_power,
                                        cluster_freq, p_model_freq,stat_sample_freq)),
                                    (power_stats->get_sfu_active_lanes(
                                        sfu_active_lanes_set_active_lanes_power,
                                        cluster_freq,p_model_freq, stat_sample_freq)),
                                    sp_active_lanes_set_active_lanes_power,
                                    sfu_active_lanes_set_active_lanes_power,
                                    stat_sample_freq);

    double *n_icnt_mem_to_simt_set_NoC_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);
    double *n_icnt_simt_to_mem_set_NoC_power =
        (double *)malloc(sizeof(double) * wrapper->number_shaders);

    double n_icnt_simt_to_mem =
        (double)
            power_stats->get_icnt_simt_to_mem(n_icnt_simt_to_mem_set_NoC_power);  // # flits from SIMT clusters
                                                                                  // to memory partitions
    double n_icnt_mem_to_simt =
        (double)
            power_stats->get_icnt_mem_to_simt(n_icnt_mem_to_simt_set_NoC_power);  // # flits from memory
                                                                                  // partitions to SIMT clusters
    wrapper->set_NoC_power(
        n_icnt_mem_to_simt,
        n_icnt_simt_to_mem,n_icnt_mem_to_simt_set_NoC_power,n_icnt_simt_to_mem_set_NoC_power);  // Number of flits traversing the interconnect





    wrapper->compute(true);

    wrapper->update_components_power(1);
    wrapper->update_components_power_per_core(0,p_model_freq);
    double Actual_power = wrapper->sum_per_sm_and_shard_power(cluster_freq);

    wrapper->print_trace_files();


    wrapper->detect_print_steady_state(0, tot_inst + inst);

    wrapper->power_metrics_calculations();

    wrapper->dump();
    //After the first interval the new frequency set is calculated but it is fed to the simulator after secoond interval
    //For the rest of the intervals, interval ith interval calculates the new frequency set for interval (i+2)th
    if(!first) {
      for (int i = 0; i < wrapper->number_shaders; i++)
        pre_cluster_freq[i] = new_cluster_freq[i];
    }
    //Available frequency levels are set here which are 100, 200, 300, 400, 500, 600, and 700 MHz
    //These levels are fed to the genetic algorithm to find the optimal frequency set 
    double * Available_freqs = (double *)malloc(sizeof(double )*7);
    for(int i=0;i<7;i++)
      Available_freqs[i] = (i+1)*100*1e6;
    unsigned I_t = 0;
    //The stall time is calculated here
    for(int i=0;i<wrapper->number_shaders;i++) {
        I_t += wrapper->I[i];
        if (m_cluster[i]->sch_cycles == 0)
            wrapper->TS[i] = 1;
        else
            wrapper->TS[i] = (double) (m_cluster[i]->waiting_warps) / 1000;
    }
    //Create the genetic algorithm class and set hyperparameters
    class Population pop = Population(10,wrapper->number_shaders,7,Available_freqs,0.2,0.2,0.4);
    //Initialize the genetic algorithm data members
    pop.mcpat_data_set(wrapper, Actual_power,cluster_freq);
    double* optimized_freq = (double*)malloc(sizeof(double)*wrapper->number_shaders);
    //Run the genetic algorithm for 10 iterations
      optimized_freq =  pop.evolution(10);
    //If there is no intructions executed, set the frequency to the lowest level
      for(int i=0;i<wrapper->number_shaders;i++){
          if(I_t == 0)
              optimized_freq[i] = 100*1e6;
          if (m_cluster[i]->sch_cycles == 0)
              optimized_freq[i] = 100*1e6;

      }
      double TS;
      double PS;
      double PC;
      double* analatical_optimized_freq = (double*)malloc(sizeof(double)*wrapper->number_shaders);
      for(int i=0;i<wrapper->number_shaders;i++){
          if( m_cluster[i]->sch_cycles == 0)
              TS = 1;
          else
              TS = (double)(m_cluster[i]->waiting_warps) / 1000;
          //Set the dynamic and static power
          PS = wrapper->sample_cmp_pwr_S[i]+wrapper->sample_cmp_pwr_Shrd/15 + wrapper->sample_cmp_pwr_const /15;
          PC = (double)59/15;
          if(TS <= 0)
              analatical_optimized_freq[i] = 700*1e6;
          else{
              if(TS >= 1 || PS == 0)
                  analatical_optimized_freq[i] = 100*1e6;
              else {
                  //Optimization using aalytical model
                  double ratio;
                  ratio = sqrt((1-TS)*PC/(PS*TS));

                  analatical_optimized_freq[i] = round(ratio * (cluster_freq[i] /(100 * 1e6))) * 100 * 1e6;

                  if (analatical_optimized_freq[i] > 700 * 1e6)
                      analatical_optimized_freq[i] = 700 * 1e6;
                  if (analatical_optimized_freq[i] < 100 * 1e6)
                      analatical_optimized_freq[i] = 100 * 1e6;
              }
          }
          std::ofstream file_stall;
          std::string path1 =  getenv("DATA_PATH");
          std::string path2 = "/DATA/ STALL/stall_";
          std::string path = path1 + path2;
          file_stall.open(path.c_str() + std::to_string(i),std::ios::app);
          file_stall<<m_cluster[i]->sch_cycles <<"\t"<<m_cluster[i]->sch_ready_inst<<"\t"<<m_cluster[i]->waiting_warps;
          file_stall.close();
          m_cluster[i]->occupancy_per_sms = 0;
          m_cluster[i]->waiting_warps = 0;
          m_cluster[i]->sch_cycles = 0;
          m_cluster[i]->sch_valid_inst = 0;
          m_cluster[i]->sch_ready_inst = 0;
      }
      double energy_GA = 0;
      double energy_anal = 0;
      double I_GA = 0;
      double I_anal = 0;
      for(int i=0;i<wrapper->number_shaders;i++) {
        energy_GA += wrapper->sample_cmp_pwr_S[i]*optimized_freq[i]/(700*1e6)+wrapper->sample_cmp_pwr_Shrd/15 + wrapper->sample_cmp_pwr_const /15 + 59/15;
          energy_anal += wrapper->sample_cmp_pwr_S[i]*analatical_optimized_freq[i]/(700*1e6)+wrapper->sample_cmp_pwr_Shrd/15 + wrapper->sample_cmp_pwr_const /15 + 59/15;
          I_GA +=  optimized_freq[i]/(700*1e6) * wrapper->I[i] / (wrapper->TS[i] * optimized_freq[i]/(700*1e6) + 1 - wrapper->TS[i]);
          I_anal +=  analatical_optimized_freq[i]/(700*1e6) * wrapper->I[i] / (wrapper->TS[i] * analatical_optimized_freq[i]/(700*1e6) + 1 - wrapper->TS[i]);
      }
        if(1) {
            for (int i = 0; i < wrapper->number_shaders; i++)
                new_cluster_freq[i] = analatical_optimized_freq[i];
        }
        else{
            for(int i=0;i<wrapper->number_shaders;i++)
                new_cluster_freq[i] = optimized_freq[i];
        }

      if (m_cluster[0]->kernel_done){
          m_cluster[0]->kernel_done = 0;
      for(int i=0;i<wrapper->number_shaders;i++)
          new_cluster_freq[i] = 700 * 1e6;

      }
      double alpha_;
      std::string path1 =  getenv("DATA_PATH");
      std::string path2 = "/DATA/STALL/stall_";
      std::string path = path1 + path2;
      for(int i=0;i<wrapper->number_shaders;i++) {
         alpha_ = (new_cluster_freq[i] / cluster_freq[i]);
          std::ofstream file_stall;
          file_stall.open(path.c_str() + std::to_string(i),std::ios::app);
          file_stall << "\t" << wrapper->I[i] << "\t" << alpha_ * wrapper->I[i] / (TS * alpha_ + 1 - TS)  << "\t" << cluster_freq[i]
                 << std::endl;
          file_stall.close();
      }
    free(n_icnt_mem_to_simt_set_NoC_power);
    free(n_icnt_simt_to_mem_set_NoC_power);
    free(sfu_active_lanes_set_active_lanes_power);
    free(tot_ins_set_inst_power);
    free(total_int_ins_set_inst_power);
    free(tot_fp_ins_set_inst_power);
    free(tot_commited_ins_set_inst_power);
    free(regfile_reads_set_regfile_power);
    free(regfile_writes_set_regfile_power);
    free(non_regfile_operands_set_regfile_power);
    free(shmem_read_set_power);
    free(active_sms_per_cluster);
    free(num_cores_per_cluster);
    free(num_idle_core_per_cluster);
    free(pipeline_duty_cycle_per_sm);
    free(tot_fpu_accessess_set_exec_unit_power);
    free(ialu_accessess_set_exec_unit_power);
    free(tot_sfu_accessess_set_exec_unit_power);
    free(sp_active_lanes_set_active_lanes_power);
    free(Available_freqs);

    free(optimized_freq);
      power_stats->save_stats(wrapper->number_shaders, numb_active_sms,average_pipeline_duty_cycle_per_sm);
      if(first) {
        first  = false;
        return false;
      }
      else
        return true;
  }
  else
    return false;
}
//Power and perfromance prediction with new frrequency set defined with genetic algorithm as fitness function
void mcpat_cycle_power_calculation(class Chromosome *chromosome,
                                   const class gpgpu_sim_wrapper *wrapper,
                                  double* base_cluster_freq){
    unsigned I = 0;
    double * I_PER_CORE = (double*) malloc(sizeof(double) * wrapper->number_shaders);
    double power = 0;
    double alpha;
    for(int i=0;i<chromosome->size;i++){
        alpha = (double)chromosome->gene[i]/base_cluster_freq[i];
        if(wrapper->I[i] == 0)
            I_PER_CORE[i] = 0;
        else {
            I += alpha * wrapper->I[i] / (wrapper->TS[i] * alpha + 1 - wrapper->TS[i]);
            I_PER_CORE[i] = alpha * wrapper->I[i] / (wrapper->TS[i] * alpha + 1 - wrapper->TS[i]);
        }
        power += wrapper->sample_cmp_pwr_S[i]*alpha+
                wrapper->sample_cmp_pwr_Shrd/15 +
                wrapper->sample_cmp_pwr_const /15;
    }
    chromosome->power = power+59;
    chromosome->power_time = power/I;
}

void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper) {
  wrapper->reset_counters();
}












