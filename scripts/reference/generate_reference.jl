### Script for generating reference data for evaluation/testing
###
### Objective: Propagate a specific model and save whole state every day for 2 years



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



### Define possible variants for reference data generation
variants = [

    # 0. Default: no longwave radiation scheme
    (;),

    # 1. OneBandLongwave default
    (; name="OBLW_default", lw_scheme=OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG))),

    # 2. AnalyticBandRadiation default
    (; name="ABR_default", lw_scheme=SpeedyExt.SpeedyAnalyticBandLongwave(SG)),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME       = get(v, :name, "default")           # name of the reference data
SEED       = get(v, :seed, 42)                  # used seed

# Model and scheme
MODEL      = get(v, :model, PrimitiveWetModel)  # used model
LW_SCHEME  = get(v, :lw_scheme, nothing)        # used longwave radiation scheme

# Sampling
T_SPINUP   = get(v, :t_spinup, Day(1))                  # spinup time in days
START_DATE = get(v, :start_date, DateTime(2000, 1, 1))  # sampling starting date
SIM_DAYS   = get(v, :sim_days, 5)                       # sampling time in days

# Perturbation
FAC_PERT_T = get(v, :fac_pert_t, 2f0)           # additive perturbation amplitude for temperature
FAC_PERT_Q = get(v, :fac_pert_q, 0.2f0)         # multiplicative perturbation amplitude for humidity (zeromin = true)



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

# Initialize simulation and perform a first timestep
SpeedyWeather.initialize!(sim, steps = SIM_DAYS * steps_per_day + 1)
SpeedyWeather.first_timesteps!(sim)



### Start data sampling
# Create container for variable states with a first entry
reference = [deepcopy(sim.variables)]


# Loop over the whole simulation
for day in 1:SIM_DAYS

    # Propagate simulation for one day
    for _ in 1:steps_per_day
        SpeedyWeather.later_timestep!(sim)
    end

    # Save state
    push!(reference, deepcopy(sim.variables))
end



### Save reference data
# Create folder
foldername = "$(NAME)"
folderpath = prepare_out_dir(reference_dir(), foldername)

# Save reference
NeuralParam.save(reference; path=folderpath, file="reference.jld2")
@info "Reference dataset stored at $(folderpath)!"

# Create and store info.toml file
write_info(; 
    path = folderpath, 
    file = "info.toml",

    name        = NAME,
    created     = now(),
    seed        = SEED,
    julia       = string(VERSION),
    sw          = string(pkgversion(SpeedyWeather)),
    
    trunc       = TRUNC,
    nlayers     = NLAYERS,

    model       = nameof(MODEL),
    lw_scheme   = nameof(typeof(LW_SCHEME)),
    
    t_spinup    = string(T_SPINUP),
    start_date  = string(START_DATE),
    sim_days    = SIM_DAYS,

    fac_pert_t  = FAC_PERT_T,
    fac_pert_q  = FAC_PERT_Q,
)
