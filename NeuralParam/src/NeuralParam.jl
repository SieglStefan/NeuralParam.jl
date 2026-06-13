module NeuralLongwave


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


export  
        # utils (without io)
        zscore,
        extract_layer,
        rmse,
        bias,
        correlation,
        maxdiff,
        plot_calibration,
        plot_training,
        plot_loss,
        plot_comparison,
        plot_heatmap,
        plot_heatmaps,
        perturb_grid_temp!,
        generate_temperature_fields,

        # parameterizations
        ConstLinearLongwave,
        NeuralLinearLongwaveConfig,
        NeuralLinearLongwave,
        NeuralABRLongwaveConfig,
        NeuralABRLongwave,

        # io
        save_neural_longwave,
        load_neural_longwave,

        # optimization
        run_calibration!,
        run_training!,
        run_optimization!



# General utils
include("utils/utils.jl")
include("utils/metrics.jl")
include("utils/plotting.jl")
include("utils/data.jl")


# Structs / Parameterizations
include("parameterizations/abstract_neural_longwave.jl")
include("parameterizations/linear_longwave/const_llw.jl")
include("parameterizations/linear_longwave/neural_llw_setup.jl")
include("parameterizations/linear_longwave/neural_llw.jl")
include("parameterizations/analytic_band_radiation/neural_abrlw_setup.jl")
include("parameterizations/analytic_band_radiation/neural_abrlw.jl")


# IO, printing and neural network utils
include("utils/io.jl")
include("utils/printing.jl")
include("utils/neural_network.jl")


# Optimizing infrastructure
include("optimization/simulation_handling.jl")
include("optimization/optimization_online.jl")
include("optimization/gradients.jl")
include("optimization/run_scheme.jl")


end
