### Script for generating rollouts for evaluation
###
### Objective: Propagate a specific model for certain horizons over a year and save states



### Load packages
using NeuralParam
using SpeedyWeather
using AnalyticBandRadiation
SpeedyExt = Base.get_extension(AnalyticBandRadiation,
                                :AnalyticBandRadiationSpeedyWeatherExt)
using Dates
using JLD2



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



models_path = joinpath(@__DIR__, "..", "..", "results", "models")
reference_path   = joinpath(@__DIR__, "..", "..", "data", "reference")

### Define possible variants for rollouts
variants = [

    # ConstLinearLW rollouts
    (; name="skill_CLLW",     reference="OneBandLongwave_default",
       max_horizon=31,  n_traj=52, heatmap_days=[1,7,31],
       schemes=[:const => joinpath(models_path, "scheme_CLLW_default")]),

    (; name="stability_CLLW", reference="OneBandLongwave_default",
       max_horizon=180, n_traj=4,  heatmap_days=[1,30,90,180],
       schemes=[:const => joinpath(models_path, "scheme_CLLW_default")]),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Load reference and schemes
reference = load(; path=joinpath(reference_path, v.reference), file="reference.jld2")
schemes   = build_schemes(v.schemes)



### Compute and save reduced diagnostics
# Rollout forecast
results = evaluate_rollout(
    schemes, reference;
    spectral_grid = SG, 
    model_type = PrimitiveWetModel,
    max_horizon = v.max_horizon, 
    n_traj = v.n_traj,
    heatmap_days = v.heatmap_days, 
    heatmap_traj = 1
)

# Save reduced diagnostics
dir = fresh_out_dir(joinpath(@__DIR__, "..", "..", "results", "rollouts"), v.name)
jldsave(joinpath(dir, "rollout.jld2"); results)

# Create and store info.toml file
write_info(; 
    path=dir, 
    file="info.toml",

    name=v.name, 
    created=now(), 
    julia=string(VERSION),
    sw=string(pkgversion(SpeedyWeather)),

    reference=v.reference, 
    max_horizon=v.max_horizon, 
    n_traj=v.n_traj,
    schemes=[string(k) for (k,_) in v.schemes], 
    
    trunc=TRUNC, 
    nlayers=NLAYERS
)