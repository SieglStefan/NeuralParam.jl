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
        NeuralLinearLongwaveAD,
        NeuralLinearLongwave,

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
include("parameterizations/abstract_longwave.jl")
include("parameterizations/const_llw.jl")
include("parameterizations/neural_llw_config.jl")
include("parameterizations/neural_llw_ad.jl")
include("parameterizations/neural_llw.jl")


# IO and printing
include("utils/io.jl")
include("utils/printing.jl")


# Training infrastructure
include("optimization/simulation_handling.jl")
include("optimization/optimization_online.jl")
include("optimization/gradients.jl")
include("optimization/run_scheme.jl")


end
