module NeuralParam


using SpeedyWeather

using Lux
using Optimisers

using Enzyme
using Checkpointing

import JLD2
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
                        #area_weights,
        # stats.jl
                        #mean_std,
                        #mean_std_layers,
                        #fit_linear,
        # tree.jl
                        #tree_l2sum,
                        #tree_l2norm,
                        #tree_add,
                        #tree_scale,
        # heatmaps.jl
                        #field_to_lonlatmat,
                        #shift_lon,
                        #finite_range,
                        #sym_range,
                plot_heatmaps,
        # simulation.jl
                steps_from_days,
                        #days_from_steps,
                perturb_grid_field!,
                first_steps!,
                sim_timesteps!,


        ### data
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
                        #resolve_scheme,
                scheme_name,
                        #_toml,
                write_info,
                prepare_out_dir,
                fresh_out_dir,
                Reference,
                with_reference,
                        #save_store,
        # zscore.jl
                        #zscore,
                        #inv_zscore,
                ZScoreStats,
                load_zscore,
                        #collect_stats,
        # generate/generate_reference.jl
                generate_reference,
        # generate/generate_rollout.jl
                DEF_PROBES,
                        #sample_trajectory,
                        #n_cols,
                        #get_col,
                generate_rollout,
        # generate/generate_zscore.jl
                        #target_outputs!,
                        #collect_samples,
                        #input_stats,
                generate_zscore,
        # generate/plot_zscore.jl
                        #plot_histograms,
                        #plot_profile_histograms,
                        #plot_scalar_histograms,
                        #thin_idx,
                        #regression_panel!,
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
                INPUTS,
                input_spec,
                INPUT_NLW_OBLW,
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
                        #mid_layer,
                        #decode!,
                        #lw_state,
                        #write_lw!,
                        #predictor,
                        #affine_stats,
                        #output_stats,
                        #output_center,
        # scheme_const.jl
                ConstLW,
                        #update_ps,
                info_scheme,
        # scheme_neural.jl
                NeuralLW,
                        #update_ps,
                        #info_scheme,
        # scheme_zero.jl
                ZeroLW,
                        #update_ps,
                        #info_scheme,

        ### training
        # config.jl
                TrainConfig,
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
                        #seed_loss,
                LossConfig,
                        #load_field_norm,
        # plotting.jl
                FIELD_COLORS,
                log_or_lin,
                ic_bounds,
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
                sample_start_date,
        # training_offline.jl
                # ---
        # training_online.jl
                        #training_online,
                        #online_gradient_step,


        ### evaluation
        # rollout.jl
                        #DEF_PROBE_UNITS,
                        #reduce_cols,
                rollout_curve,
                plot_rollout,
                plot_rollout_metrics,
                        #hm_field,
                plot_rollout_heatmaps,
        # benchmark.jl
                evaluate_benchmark,
                print_benchmark





# General utils
include("utils/metrics.jl")
include("utils/stats.jl")
include("utils/tree.jl")
include("utils/heatmaps.jl")
include("utils/simulation.jl")


# Data generation
include("data/io.jl")
include("data/zscore.jl")
include("data/generate/generate_reference.jl")
include("data/generate/generate_rollout.jl")
include("data/generate/generate_zscore.jl")
include("data/generate/plot_zscore.jl")


# Architectures
include("architectures/abstract_arch.jl")
include("architectures/mlp.jl")
include("architectures/rnn.jl")


# Parameterizations
include("parameterizations/abstract.jl")
include("parameterizations/longwave/abstract_longwave.jl")
include("parameterizations/longwave/input.jl")
include("parameterizations/longwave/output.jl")
include("parameterizations/longwave/scheme_zero.jl")
include("parameterizations/longwave/scheme_const.jl")
include("parameterizations/longwave/scheme_neural.jl")


# Training infrastructure
include("training/loss.jl")
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
