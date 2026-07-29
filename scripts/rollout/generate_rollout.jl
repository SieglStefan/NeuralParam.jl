### Script for generating rollouts for evaluation
###
### Objective: Propagate a specific model for certain horizons over a year and save states



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using AnalyticBandRadiation
SpeedyExt = Base.get_extension(AnalyticBandRadiation,
                                :AnalyticBandRadiationSpeedyWeatherExt)
using Dates
using Random



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



### Define common parameters for task selection
# Schemes
OBLW        = OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG))
ABR         = SpeedyExt.SpeedyAnalyticBandLongwave(SG)

# References
REF_OBLW    = "OBLW_default"
REF_ABR     = "ABR_default"

# Trajectory
SKILL       = (; max_horizon = 31,  n_traj = 52, heatmap_days = [1,3,7,14])
STAB        = (; max_horizon = 180, n_traj = 4,  heatmap_days = [1,30,90,180])




### Define possible variants for rollouts
variants = [

    ### 0. Default - no used
    (;),


    ### FreeRun OneBandLongwave (no longwave parameterization)
    # 1. Skill
    (;  name            = "FreeRun_default_skill",
        scheme          = nothing,
        reference       = REF_OBLW,
        SKILL...
    ),
    # 2. Stability
    (;  name            = "FreeRun_default_stab",
        scheme          = nothing,
        reference       = REF_OBLW,
        STAB...
    ),

    
    ### Default OneBandLongwave (for sanity test, should lead to zero error in evaluation)
    # 3. Skill
    (;  name            = "OBLW_default_skill",
        scheme          = OBLW,
        reference       = REF_OBLW,
        SKILL...
    ),
    # 4. Stability
    (;  name            = "OBLW_default_stab",
        scheme          = OBLW,
        reference       = REF_OBLW,
        STAB...
    ),


    ### Default ConstLinearLW rollouts
    # 5. Skill
    (;  name            = "CLLW_default_skill",
        scheme          = "CLLW_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 6. Stability
    (;  name            = "CLLW_default_stab",
        scheme          = "CLLW_default",
        reference       = REF_OBLW,
        STAB...
    ),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME         = get(v, :name, "NoName_default")      # name of the rollout

# Scheme and reference
SCHEME       = get(v, :scheme, "CLLW_default")      # scheme to
REFERENCE    = get(v, :reference, "OBLW_default")   # reference for comparison

# Model
MODEL        = get(v, :model, PrimitiveWetModel)    # used model
VARS         = get(v, :vars, (:temperature,))       # sampled variables

# Rollout
MAX_HORIZON  = get(v, :max_horizon, 31)             # maximum forecast length in days
N_TRAJ       = get(v, :n_traj, 52)                  # number of trajectories sampled

# Metrics
METRICS      = get(v, :metrics, (; rmse, bias))     # metrics to be calculated

# Heatmaps
HEATMAP_DAYS = get(v, :heatmap_days, [1,7,31])      # days for which entire temperature fields are returned
HEATMAP_TRAJ = get(v, :heatmap_traj, 1)             # choice of specific trajectory for heatmap



### Load to be rolled out scheme and reference
scheme = resolve_scheme(SCHEME)
reference = NeuralParam.load(; path=reference_dir(REFERENCE), file="reference.jld2")



### Compute and save reduced diagnostics
# Extract time stepping
Δt_sec = initialize!(MODEL(SG; longwave_radiation = scheme)).model.time_stepping.Δt_sec

# Calculate number of steps for one day
n_steps_day = steps_from_days(1, Δt_sec)

# Calculate starting days
start_days = round.(Int, range(0, 365, length = N_TRAJ + 1))[1:end-1]


# Prepare container for metrics and heatmap data
sums = (; (var => map(_ -> zeros(Float64, MAX_HORIZON), METRICS) for var in VARS)...)
kept = nothing


# Loop over all starting days in a year e.g.: (0,7,14,...,350,364)
for (i,s) in enumerate(start_days)

    # Initialize simulation
    sim  = initialize!(MODEL(SG; longwave_radiation = scheme))


    # Sample trajectory
    traj = sample_grid_trajectory(
        sim,
        VARS, 
        reference[s + 1]; 
        n_steps = MAX_HORIZON * n_steps_day,
        n_gap = n_steps_day
    )

    # Throw error if varaible not available
    isnothing(traj) && error("Trajectory sampling failed: VARS = $VARS not available in $(MODEL).")


    # Loop over horizons
    for h in 1:MAX_HORIZON

        # Loop over variables
        for var in VARS
            f = traj.[var][h + 1]                           # sampled trajectroy 
            t = getfield(reference[s+h+1].grid, var)        # reference at day s+h
                
            # Calculate metric and add to sums
            for (name, metric) in pairs(METRICS)
                sums[var][name][h] += metric(f,t)
            end
        end
    end

    # Filter specific heatmap days
    if i == HEATMAP_TRAJ
        global kept = (; (var => [traj[var][d+1] for d in HEATMAP_DAYS] for var in VARS)...)
    end
end

# Calculate mean in respect to all starting days
rollout_stats = map(metrics -> map(c -> c ./ length(start_days), metrics), sums)


# Collet rollout results
rollout = (;
    days            = 1:MAX_HORIZON,
    vars            = VARS,
    curve           = rollout_stats,
    heatmap_days    = HEATMAP_DAYS,
    heatmap_states  = kept, 
)



### Save reduced rollout
# Create folder
foldername = "$(NAME)"
folderpath = fresh_out_dir(rollout_dir(), foldername)

# Save rollout
NeuralParam.save(rollout; path=folderpath, file="rollout.jld2")
@info "Rollout dataset stored at $(folderpath)!"

# Create and store info.toml file
write_info(; 
    path = folderpath, 
    file = "info.toml",

    name         = NAME, 
    created      = now(), 
    julia        = string(VERSION),
    sw           = string(pkgversion(SpeedyWeather)),

    trunc        = TRUNC, 
    nlayers      = NLAYERS,

    model        = MODEL,
    vars         = [string(v) for v in VARS],

    scheme       = scheme_name(SCHEME),
    reference    = REFERENCE,

    max_horizon  = MAX_HORIZON,
    n_traj       = N_TRAJ,

    metrics      = [string(k) for k in keys(METRICS)],

    heatmap_days = HEATMAP_DAYS,
    heatmap_traj = HEATMAP_TRAJ,
)