module NeuralLongwave

using SpeedyWeather
using Lux
using Optimisers
using Enzyme

using Random
using JLD2
using Plots

export LinearLongwave,
       NeuralLinearLongwave,
       NeuralLinearLongwaveConfig,
       create_sim_pair,
       sim_pair_timestep!,
       sim_pair_pullback!,
       compute_gradients,
       run_calibration!,
       run_training!,
       calibration_step!,
       training_step!,
       perturb_grid_temp!,
       MSE,
       zscore,
       plot_calibration,
       plot_loss,
       plot_rmse_diff,
       plot_bias_diff,
       plot_correlation_diff,
       save,
       load_neural_longwave



# General utils
include("utils/utils.jl")
include("utils/metrics.jl")
include("utils/plotting.jl")
include("utils/perturb_vars.jl")


# Structs / Parameterizations
include("parameterizations/linear_longwave.jl")
include("parameterizations/neural_linear_longwave.jl")


# IO
include("utils/io.jl")


# Training infrastructure
include("training/simulation.jl")
include("training/online_optimization.jl")
include("training/gradients.jl")
include("training/run_calibration.jl")
include("training/run_training.jl")


end
