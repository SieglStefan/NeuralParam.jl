### Script for generating reference data for evaluation/testing
###
### Objective: Propagate a specific model and save state every day for a year




### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Dates
using Random


### Define simulation parameters
# General
NAME = ""
CREATED = now()
SEED = 42

# Grid
TRUNC = 31
NLAYERS = 8

# Model
MODEL = PrimitiveWetModel
TRANS = FriersonLongwaveTransmissivity

# Sampling
T_SPINUP   = Day(31)
START_DATE = DateTime(2000, 1, 1)    
SIM_DAYS   = 2*365

# Perturbation
FAC_PERT_T = 2f0
FAC_PERT_Q = 0.2f0





### Prepare simulation
# Set seed for reproducability
Random.seed!(SEED)

# Define spectral grid, target scheme and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
lw_scheme = OneBandLongwave(spectral_grid; TRANS(spectral_grid))
model = MODEL(spectral_grid; longwave_radiation = lw_scheme)

# Initialize simulation
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


# Create container for variable states with a first entry
states = [deepcopy(sim.variables)]


# Loop over the whole simulation
for day in 1:SIM_DAYS

    # Propagate simulation for one day
    for _ in 1:steps_per_day
        SpeedyWeather.later_timestep!(sim)
    end

    push!(states, deepcopy(sim.variables))
end


# Folder for data
foldername = "data_L$(NLAYERS)_T$(TRUNC)_$(MODEL)_$(NAME)"
folderpath = joinpath(@__DIR__, "..", "..", "data", "evaluation", foldername)
mkpath(folderpath)

# Save states
save_reference(states; path = folderpath, file = "data.jld2")


# Create and store meta data .toml file
write_info(; 
    path = folderpath, 
    file = "meta.toml",

    name = NAME,
    created = CREATED,
    seed = SEED,
    julia= string(VERSION),
    
    trunc          = TRUNC,
    nlayers        = NLAYERS,

    model_type     = nameof(MODEL),
    lw_scheme         = nameof(typeof(lw_scheme)),
    transmissivity = nameof(typeof(TRANS)),
    
    t_spinup       = string(T_SPINUP),
    start_date     = string(START_DATE),
    sim_days       = SIM_DAYS,

    amp_t = FAC_PERT_T,
    amp_q = FAC_PERT_Q,
)
