### ConstLinearLW Evaluation
###
### XXX



### Load packages
using NeuralParam
using SpeedyWeather
using Plots
using CairoMakie
using Random
using Dates
using Statistics
using BenchmarkTools



### Parameters
# General
SEED = 42
EVA_NAME = get(ENV, "EVA_NAME", "")

# Grid
TRUNC = 31                                  #
NLAYERS = 8                                 #

# Model
MODEL = PrimitiveWetModel                   #

# Reference
LW_SCHEME = OneBandLongwave                 #
TRANS = FriersonLongwaveTransmissivity      #

# Perturbation
FAC_PERT_T = 2f0                            #
FAC_PERT_Q = 0.2f0                          #



### General
# Set seed
Random.seed!(SEED)

# Spectral grid
sg = SpectralGrid(trunc=TRUNC, nlayers = NLAYERS)

# Saving 
run_dir = joinpath(@__DIR__, EVA_NAME)
mkpath(run_dir)

# Load reference
REF = load_reference() # XXX

# Task selection
PART = get(ENV, "TASK", "")
should_run(i) = isempty(PART) || parse(Int, PART) == i

# Calibration dir
RESULTS_DIR = joinpath(@__DIR__, "..", "..")



# ----------------------------------- START EVALUATION ----------------------------------- #

### Skill Evaluation
if should_run(1)
    # Parameters
    MAX_HORIZON_SKILL = 31
    N_TRAJ_SKILL = 52
    HEATMAP_DAYS_SKILL = []
    HEATMAP_TRAJ_SKILL = 1
    HEATMAP_LAYER_SKILL = [NLAYERS],

    # Schemes
    SCHEMES_SKILL = build_schemes([
        :const     => joinpath(RESULTS_DIR,  "1_ConstLinearLW",  "calibration", "calib_X", "run_X"),
        :neural    => joinpath(RESULTS_DIR,  "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"),
        :neural_t3 => (joinpath(RESULTS_DIR, "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"), 3),
        :target    => nothing,
        :none      => nothing,
    ])

    # Run evaluation
    run_evaluation_forecast(
        schemes = SCHEMES_SKILL,
        reference = REF;
        spectral_grid = sg,
        model_type = MODEL,
        max_horizon = MAX_HORIZON_SKILL,
        n_traj = N_TRAJ_SKILL,
        heatmap_days = HEATMAP_DAYS_SKILL,
        heatmap_traj = HEATMAP_TRAJ_SKILL,
        heatmap_layer = HEATMAP_LAYER_SKILL,
        run_dir = run_dir,
        folder_name = "skill",
    )  

    # Write .toml info
    write_info(;
        path = joinpath(run_dir, "skill"),
        file = "params.toml",
        max_horizon  = MAX_HORIZON_SKILL,
        n_traj       = N_TRAJ_SKILL,
        heatmap_days = HEATMAP_DAYS_SKILL,
        heatmap_traj = HEATMAP_TRAJ_SKILL,
        schemes      = [string(k) for k in keys(SCHEMES_SKILL)],
    )
end



### Stability Evaluation
if should_run(2)
    # Parameters
    MAX_HORIZON_STAB = 180
    N_TRAJ_STAB = 4
    HEATMAP_DAYS_STAB = [1,7,31,90]
    HEATMAP_TRAJ_STAB = 1
    HEATMAP_LAYER_STAB = [NLAYERS]

    # Schemes
    SCHEMES_STAB = build_schemes([
        :const     => joinpath(RESULTS_DIR,  "1_ConstLinearLW",  "calibration", "calib_X", "run_X"),
        :neural    => joinpath(RESULTS_DIR,  "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"),
        :neural_t3 => (joinpath(RESULTS_DIR, "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"), 3),
        :target    => nothing,
        :none      => nothing,
    ])

    # Run evaluation
    run_evaluation_forecast(
        schemes = SCHEMES_STAB,
        reference = REF;
        spectral_grid = sg,
        model_type = MODEL,
        max_horizon = MAX_HORIZON_STAB,
        n_traj = N_TRAJ_STAB,
        heatmap_days = HEATMAP_DAYS_STAB,
        heatmap_traj = HEATMAP_TRAJ_STAB,
        heatmap_layer = HEATMAP_LAYER_STAB,
        run_dir = run_dir,
        folder_name = "stability",
    )  

    # Write .toml info
    write_info(;
        path = joinpath(run_dir, "stability"),
        file = "params.toml",
        max_horizon  = MAX_HORIZON_STAB,
        n_traj       = N_TRAJ_STAB,
        heatmap_days = HEATMAP_DAYS_STAB,
        heatmap_traj = HEATMAP_TRAJ_STAB,
        schemes      = [string(k) for k in keys(SCHEMES_STAB)],
    )
end



### Benchmark Evaluation
if should_run(3)
    # Parameters
    N_STEPS = 180

    # Schemes
    SCHEMES_BENCH = build_schemes([
        :const     => joinpath(RESULTS_DIR,  "1_ConstLinearLW",  "calibration", "calib_X", "run_X"),
        :neural    => joinpath(RESULTS_DIR,  "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"),
        :neural_t3 => (joinpath(RESULTS_DIR, "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"), 3),
        :target    => nothing,
        :none      => nothing,
    ])

    # Run evaluation
    run_evaluation_benchmark(
        schemes = SCHEMES_BENCH;
        spectral_grid = sg,
        model_type = MODEL,
        n_steps = N_STEPS,
        run_dir = run_dir,
    )

    # Write .toml info
    write_info(;
        path = joinpath(run_dir, "benchmark"),
        file = "params.toml",
        n_steps = N_STEPS,
        schemes = [string(k) for k in keys(SCHEMES_BENCH)],
    )   
end



### ab Evaluation
if should_run(4)
    # Parameters
    # -

    # Schemes
    SCHEMES_AB = build_schemes([
        :const     => joinpath(RESULTS_DIR,  "1_ConstLinearLW",  "calibration", "calib_X", "run_X"),
        :neural    => joinpath(RESULTS_DIR,  "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"),
        :neural_t3 => (joinpath(RESULTS_DIR, "2_NeuralLinearLW", "calibration", "calib_Y", "run_Y"), 3),
        :target    => nothing,
        :none      => nothing,
    ])

    # Run evaluation
    run_evaluation_ab(
        schemes = SCHEMES_AB;
        nlayers = NLAYERS,
        run_dir = run_dir
    )

    # Write .toml info
    write_info(;
        path = joinpath(run_dir, "ab"),
        file = "params.toml",
        schemes = [string(k) for k in keys(SCHEMES_AB)],
    )
end

# ----------------------------------- END EVALUATION ----------------------------------- #