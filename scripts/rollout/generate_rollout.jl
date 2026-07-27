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


















function evaluate_rollout(
    scheme,                                     # evaluated scheme
    reference;                                  # comparison reference
    spectral_grid,          
    model_type,             # used model for schemes
    max_horizon,                           # maximum forecast length in days
    n_traj,                                 # number of trajectories sampled
    metrics::NamedTuple = (; rmse, bias),       # used metrics
    heatmap_days,                          # list of days data for heatmaps is returned
    heatmap_traj,                           # choice of specific trajectory
)

    # Extract time stepping
    Δt_sec = initialize!(model_type(spectral_grid; longwave_radiation = scheme)).model.time_stepping.Δt_sec

    # Calculate number of steps for one day
    n_steps_day = steps_from_days(1, Δt_sec)

    # Calculate starting days
    start_days = round.(Int, range(0, 365, length = n_traj + 1))[1:end-1]


    # Prepare container for metrics
    sums = map(_ -> zeros(Float64, max_horizon), metrics)       # = (; rmse=[0,0,...],  bias=[0,0,...], ...)

    # Prepare container for heatmap data
    kept_traj = nothing


    # Loop over all starting days in a year e.g.: (0,7,14,...)
    for (i,s) in enumerate(start_days)

        # Initialize simulation
        sim  = initialize!(model_type(spectral_grid; longwave_radiation = scheme))

        # Sample trajectory
        traj = sample_trajectory(
            sim, 
            reference[s + 1]; 
            n_steps = max_horizon * n_steps_day,
            n_gap = n_steps_day                     # samplee once every day
        )


        # Loop over horizons
        for h in 1:max_horizon
            f = traj.temperature[h + 1]                     # sampled trajectroy temperature
            t = reference[s + h + 1].grid.temperature       # reference temperature at day s+h
            
            # Calculate metric and add to sums
            for (name, metric) in pairs(metrics)
                sums[name][h] += metric(f, t)
            end
        end

        # Filter heatmap days
        if i == heatmap_traj
            kept_traj = [traj.temperature[d+1] for d in heatmap_days]
        end
    end

    # Calculate mean in respect to all starting days
    rollout_stats = map(v -> v ./ length(start_days), sums)


    return (; days = 1:max_horizon, curve = rollout_stats, heatmaps = kept_traj)
end



















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