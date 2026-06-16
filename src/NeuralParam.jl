module NeuralParam


using SpeedyWeather

using Lux
using Optimisers

using Enzyme
using Checkpointing

using Random
using JLD2
using Dates
using Statistics

using Plots
using CairoMakie
using GeoMakie
using RingGrids

using Accessors


export  
        ### utils
        # utils.jl
                extract_layer,
        # stats.jl
                zscore,
                inv_zscore,
                ZScoreStats,
                load_zscore,
                load_output_scaling,
        # metrics.jl
                rmse,
                bias,
                correlation,
                maxdiff,
                tree_l2sum,
                tree_l2norm,
        # io.jl
                save_longwave,
                load_longwave,
        # printing.jl
                print_ic,
                print_traj,
                print_epoch,
                print_config,      
        # plotting.jl
                # XXX
        # data.jl
                perturb_grid_field!,

        ### parameterizations
        # ConstLinearLW
                ConstLinearLWConfig,
                ConstLinearLW,
        # NeuralLinearLW
                NeuralLinearLWConfig,
                NeuralLinearLW,
        # NeuralABRLWConfig
                NeuralABRLWConfig,
                NeuralABRLW,

        ### models
        #mlp.jl
                MLPConfig,
                setup_nn,

        ### training
        # gradients.jl
                compute_gradients,
                checkpointed_timesteps!,
                seed_loss!,
        # training_config.jl
                TrainingConfig,
        # run_training.jl
                run_training,
        # training_offline
                # XXX
        # training_online
                training_online,
                online_training_step,
                update_ps,
                sim_timesteps!





# General utils
include("utils/utils.jl")
include("utils/stats.jl")
include("utils/metrics.jl")
include("utils/io.jl")
include("utils/printing.jl")
include("utils/plotting.jl")
include("utils/data.jl")



# Models
include("models/abstract_nn.jl")
include("models/mlp.jl")
include("models/rnn.jl")



# Parameterizations
include("parameterizations/longwave/abstract_longwave.jl")
include("parameterizations/longwave/linear/const_linear_config.jl")
include("parameterizations/longwave/linear/const_linear.jl")
include("parameterizations/longwave/linear/neural_linear_config.jl")
include("parameterizations/longwave/linear/neural_linear.jl")
include("parameterizations/longwave/abr/neural_abr_config.jl")
include("parameterizations/longwave/abr/neural_abr.jl")



# Training infrastructure
include("training/gradients.jl")
include("training/training_config.jl")
include("training/training_online.jl")
include("training/training_offline.jl")
include("training/run_training.jl")


end
