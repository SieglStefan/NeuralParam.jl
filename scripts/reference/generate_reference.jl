### Script for generating reference data for evaluation/testing
###
### Objective: Propagate a specific model and save state every day for 2 years



### Load packages
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



### Define possible variants for reference data generation
variants = [

    # Default: no longwave radiation scheme
    (;),

    # OneBandLongwave default
    (; name="OneBandLongwave_default", lw_scheme=OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG))),

    # AnalyticBandRadiation default
    (; name="AnalyticBandRadiation_default", lw_scheme=SpeedyExt.SpeedyAnalyticBandLongwave(SG)),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME =       get(v, :name, "default")
SEED =       get(v, :seed, 42)

# Model and scheme
MODEL =      get(v, :model, PrimitiveWetModel)
LW_SCHEME =  get(v, :lw_scheme, nothing)

# Sampling
T_SPINUP   = get(v, :t_spinup, Day(31))
START_DATE = get(v, :start_date, DateTime(2000, 1, 1))    
SIM_DAYS   = get(v, :sim_days, 30)

# Perturbation
FAC_PERT_T = get(v, :fac_pert_t, 2f0)
FAC_PERT_Q = get(v, :fac_pert_q, 0.2f0)



### Prepare simulation
# Set seed for reproducability
Random.seed!(SEED)

# Define model and initialize simulation
model = MODEL(SG; longwave_radiation = LW_SCHEME)
sim = initialize!(model)


# Set starting time for spinup
clock_start = START_DATE - T_SPINUP
SpeedyWeather.set!(sim.variables.prognostic.clock; time = clock_start, start = clock_start)

# Perturb grid fields
perturb_grid_field!(sim, :temperature; fac_add  = FAC_PERT_T)
perturb_grid_field!(sim, :humidity;    fac_mult = FAC_PERT_Q, zeromin = true)

# Spinup simulation
run!(sim, period = T_SPINUP)

# Extract time-step time and calculate necessary steps for one day
(; Δt_sec) = sim.model.time_stepping
steps_per_day = steps_from_days(1, Δt_sec)


SpeedyWeather.initialize!(sim, steps = SIM_DAYS * steps_per_day)



### Start data sampling
# Create container for variable states with a first entry
states = [deepcopy(sim.variables)]


# Loop over the whole simulation
for day in 1:SIM_DAYS

    # Propagate simulation for one day
    for _ in 1:steps_per_day
        SpeedyWeather.time_step!(sim)
    end

    push!(states, deepcopy(sim.variables))
end



### Save reference data
# Create folder
base       = joinpath(@__DIR__, "..", "..", "data", "reference")
foldername = "data_L$(NLAYERS)_T$(TRUNC)_$(MODEL)_$(NAME)"
folderpath = prepare_out_dir(base, foldername)

# Save reference
save(states; path = folderpath, file = "reference.jld2")

# Create and store info.toml file
write_info(; 
    path = folderpath, 
    file = "info.toml",

    name            = NAME,
    created         = now(),
    seed            = SEED,
    julia           = string(VERSION),
    sw              = string(pkgversion(SpeedyWeather)),
    
    trunc           = TRUNC,
    nlayers         = NLAYERS,

    model      = nameof(MODEL),
    lw_scheme       = nameof(typeof(LW_SCHEME)),
    
    t_spinup        = string(T_SPINUP),
    start_date      = string(START_DATE),
    sim_days        = SIM_DAYS,

    amp_t           = FAC_PERT_T,
    amp_q           = FAC_PERT_Q,
)
