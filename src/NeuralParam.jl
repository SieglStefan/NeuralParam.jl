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
        # metrics.jl
                        #wmean,
                        #rmse,
                        #bias,
                        #wrmse,
                        #wbias,
                        #correlation,
                        #maxdiff,
                        #tree_l2sum,
                        #tree_l2norm,
                        #tree_add,
                        #tree_scale,
        # plotting.jl
                        #field_to_lonlatmat,
                        #shift_lom,
                        #finite_range,
                plot_heatmap,
                plot_heatmaps,
        # utils.jl
                extract_layer,
                steps_from_days,
                days_from_steps,


        ### data
        # data_gen.jl
                perturb_grid_field!,
                sample_start_date,
                first_steps!,
                sim_timesteps!,
                PROBES,
                sample_trajectory,
                sample_grid_trajectory,
        # io.jl
                        #save,
                        #load,
                        #ROOT,
                reference_dir,
                stats_dir,
                scheme_dir,
                rollout_dir,
                collect_schemes,
                collect_rollouts,
                resolve_scheme,
                scheme_name,
                        #_toml,
                write_info,
                prepare_out_dir,
                fresh_out_dir,
                Reference,
                        #restart_days,
                grid_state,
                verify_state,
                with_reference,
                save_store,
        # zscore.jl
                        #zscore,
                        #inv_zscore,
                        #mean_std,
                        #mean_std_layers,
                        #fit_linear,
                ZScoreStats,
                load_zscore,
                        #collect_stats,
        # generate_reference.jl
                generate_reference,
        # generate_rollout.jl
                        #n_cols,
                        #get_col,
                generate_rollout,
        # generate_zscore.jl
                        #target_outputs!,
                        #collect_samples,
                        #subsample,
                        #input_stats,
                generate_zscore,
        # plot_zscore.jl
                        #plot_histograms,
                        #plot_profile_histograms,
                        #plot_scalar_histograms,
                        #thin,
                        #plot_regression_dT,
                        #plot_regression_flux,
                plot_zscore,



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
                        #AbstractLW,
        # input.jl
                        #in_T,
                        #in_q,
                        #in_p,
                        #in_lat,
                        #in_lf,
                        #in_sst,
                        #in_lst,
                        #in_Ts,
                        #INPUTS,
                        #input_spec,
                IN_NLW_OBLW,
                        #n_inputs,
                        #input_layout,
                        #fill_inputs!,
                        #surface_temp,
        #output.jl
                LinearOutput,
                DirectOutput,
                PlanckOutput,
                        #n_outputs,
                output_group,
                output_keys,
                        #olw_layer,
                        #decode!,
                        #lw_state,

                        #write_lw!,
                        #predictor,
                        #affine_stats,
                        #output_stats,
        # scheme_const.jl
                ConstLW,
                        #update_ps,
                info_scheme,
        # scheme_neural.jl
                NeuralLW,

        ### training
        # config.jl
                RunConfig,
        # gradients.jl
                        #compute_gradients,
                        #checkpointed_timesteps!,
        # logging.jl
                        #csv_init,
                        #csv_row!,
                        #csv_read,
                        #compute_metrics,
                print_config,
        # loss.jl
                        #seed_loss!,
                LossConfig,
                        #area_weights,
                        #load_field_norm,
        # plotting.jl
                plot_loss,
                plot_training,
                plot_training_comp,
                plot_metrics_norm,
                plot_metrics_raw,
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
                        #reduce_cols,
                rollout_curve,
                plot_rollout,
                        #hm_field,
                        #sym_range,
                plot_rollout_heatmaps,
        # benchmark.jl
                evaluate_benchmark,
                print_benchmark





# General utils
include("utils/utils.jl")
include("utils/metrics.jl")
include("utils/plotting.jl")


# Data generation
include("data/io.jl")
include("data/data_gen.jl")
include("data/zscore.jl")
include("data/generate_reference.jl")
include("data/generate_rollout.jl")
include("data/generate_zscore.jl")
include("data/plot_zscore.jl")


# Architectures
include("architectures/abstract_arch.jl")
include("architectures/mlp.jl")
include("architectures/rnn.jl")


# Parameterizations
include("parameterizations/abstract.jl")
include("parameterizations/longwave/abstract_longwave.jl")
include("parameterizations/longwave/input.jl")
include("parameterizations/longwave/output.jl")
include("parameterizations/longwave/scheme_const.jl")
include("parameterizations/longwave/scheme_neural.jl")


# Training infrastructure
include("training/loss.jl")         # must precede config.jl: RunConfig annotates a LossConfig field
include("training/config.jl")
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
