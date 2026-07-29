module NeuralParam


using SpeedyWeather

using Lux
using Optimisers

using Enzyme
using Checkpointing

using JLD2
using CSV
using DataFrames
using TOML

using Random
using Dates
using Statistics

using Plots
using CairoMakie
using GeoMakie
using RingGrids

using Accessors

using Adapt
using MLDataDevices: cpu_device, gpu_device
using BenchmarkTools



export  
        ### utils
        # data.jl
                perturb_grid_field!,
                sim_timesteps!,
                sample_grid_trajectory,
                
        # io.jl
                ROOT,
                reference_dir,
                stats_dir,
                scheme_dir,
                rollout_dir,
                collect_schemes,
                collect_rollouts,
                load_stats,
                resolve_scheme,
                scheme_name,
                        #save,
                        #load,
                        #_toml,
                write_info,
                prepare_out_dir,
                fresh_out_dir,
                Reference,
                grid_state,
                with_reference,
                save_store,
        # metrics.jl
                        #mse,
                rmse,
                bias,
                        #correlation,
                        #maxdiff,
                        #tree_l2sum,
                        #tree_l2norm,
                        #tree_add,
                        #tree_scale,
        # plotting.jl
                        #field_to_lonlatmat,
                plot_heatmap,
                plot_heatmaps,
                plot_histograms,
        # stats.jl
                Scaling,
                        #to_cpu,
                zscore,
                inv_zscore,
                ZScoreStats,     
        # utils.jl
                extract_layer,
                steps_from_days,
                days_from_steps,
                        #sample_start_date,
                        #target_colorrange,
                        #area_weights,

        ### architectures
        # abstract_arch.jl
                        #AbstractArchConfig,
        # mlp.jl
                MLPConfig,
                        #setup_arch,
                        #info_arch,
        # rnn.jl
                # ---

        ### parameterizations
        # abstract_longwave.jl
                #AbstractNeuralLW,
                #AbstractLinearLW,
                #AbstractABRLW,
                #to_cpu,
        ## /linear       
        # const_linear.jl
                ConstLinearLW,
                        #update_ps,
                info_scheme, 
        # neural_linear.jl
                NeuralLinearLW,
                warmstart!,
                        # (-//-)
        ## /abr
        # neural_abr.jl
                NeuralABRLW,
                        # (-//-)
        # neural_abr_global.jl
                NeuralABRLWGlobal,
                        # (-//-)

        ### training
        # config.jl
                RunConfig,
                OutputConfig,
        # gradients.jl
                        #compute_gradients,
                        #checkpointed_timesteps!,
        # logging.jl
                        #csv_init,
                        #csv_row!,
                        #csv_read,
                        #print_ic,
                        #print_traj,
                print_config,
        # loss.jl
                        #compute_metrics,
                        #seed_loss!,

        # setup.jl
                        #setup_simulations,
                        #setup_optimiser,
                        #prepare_reference,
        # plotting.jl
                plot_loss,
                plot_training,
                plot_training_comp,
                plot_metrics,
        # run_training.jl
                run_training,
        # setup.jl
                        #setup_simulations,
                        #setup_optimiser,
                        #prepare_reference,
        # training_offline.jl
                # ---
        # training_online.jl
                        #training_online,
                        #online_training_step,


        ### evaluation
        # rollout.jl
                plot_rollout,
        # benchmark.jl
                evaluate_benchmark,
                print_benchmark





# General utils
include("utils/utils.jl")
include("utils/metrics.jl")
include("utils/data.jl")
include("utils/io.jl")
include("utils/plotting.jl")
include("utils/stats.jl")


# Architectures
include("architectures/abstract_arch.jl")
include("architectures/mlp.jl")
include("architectures/rnn.jl")


# Parameterizations
include("parameterizations/longwave/abstract_longwave.jl")
include("parameterizations/longwave/linear/const_linear.jl")
include("parameterizations/longwave/linear/neural_linear.jl")
include("parameterizations/longwave/abr/neural_abr.jl")
include("parameterizations/longwave/abr/neural_abr_global.jl")


# Training infrastructure
include("training/config.jl")
include("training/loss.jl")
include("training/gradients.jl")
include("training/setup.jl")
include("training/plotting.jl")
include("training/logging.jl")
include("training/training_online.jl")
include("training/training_offline.jl")
include("training/run_training.jl")


# Evaluation infrastructure
include("evaluation/rollout.jl")
include("evaluation/benchmark.jl")


end
