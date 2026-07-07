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
using CUDA
using cuDNN
using MLDataDevices: cpu_device, gpu_device


export  
        ### utils
        # data.jl
                perturb_grid_field!,
        # device.jl
                        #to_cpu,
        # io.jl
                save_scheme,
                load_scheme,
                        #load_stats,
                        #csv_init,
                        #csw_row!,
                csv_info,
                        #csv_read,
                        #arch_meta,
                        #meta_scheme,
                        #build_meta,
                        #_toml,
                write_info,
        # metrics.jl
                mse,
                rmse,
                bias,
                correlation,
                maxdiff,
                tree_l2sum,
                tree_l2norm,
        # plotting.jl
                plot_loss,
                plot_training,
                plot_training_comp,
                plot_metrics,
                plot_comparison,
                        #field_to_lonlatmat,
                plot_heatmap,
                plot_heatmaps,
                plot_histograms,
        # printing.jl
                        #print_ic,
                        #print_traj,
                        #print_epochs,
                print_config,    
        # stats.jl
                Scaling,
                zscore,
                inv_zscore,
                ZScoreStats,
        # utils.jl
                extract_layer,


        ### architectures
        # abstract_arch.jl
                        #AbstractArchConfig,
        # mlp.jl
                MLPConfig,
                        #setup_arch,
        # rnn.jl
                # ---

        ### parameterizations
        # const_linear.jl
                ConstLinearLW,
        # neural_linear.jl
                NeuralLinearLW,
        # neural_abr.jl
                NeuralABRLW,
        # neural_abr_global.jl
                NeuralABRLWGlobal,


        ### training
        # config.jl
                RunConfig,
                OutputConfig,
        # loss.jl
                compute_metrics,
                        #seed_loss!,
        # gradients.jl
                        #compute_gradients,
                        #checkpointed_timesteps!,
                        
        # run_training.jl
                run_training
        # training_offline
                # ---
        # training_online
                        #training_online,
                        #online_training_step,
                        #update_ps,
                        #sim_timesteps!





# General utils (expect io, printing and device management)
include("utils/utils.jl")
include("utils/stats.jl")
include("utils/metrics.jl")
include("utils/plotting.jl")
include("utils/data.jl")


# Models
include("architectures/abstract_arch.jl")
include("architectures/mlp.jl")
include("architectures/rnn.jl")


# Parameterizations
include("parameterizations/longwave/abstract_longwave.jl")
include("parameterizations/longwave/linear/const_linear.jl")
include("parameterizations/longwave/linear/neural_linear.jl")
include("parameterizations/longwave/abr/neural_abr.jl")
include("parameterizations/longwave/abr/neural_abr_global.jl")


# IO, printing and device management
include("utils/io.jl")
include("utils/printing.jl")
include("utils/device.jl")


# Training infrastructure
include("training/loss.jl")
include("training/gradients.jl")
include("training/config.jl")
include("training/training_online.jl")
include("training/training_offline.jl")
include("training/run_training.jl")


end
