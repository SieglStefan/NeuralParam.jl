### Script for generating reference data for evaluation/testing
###
### Objective: Propagate a specific model and save its state every day for 2 years



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Dates
using Random



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



### Define possible variants for reference data generation
variants = [

    # 0. Test variant for quick testing  
    (;  name        = "TEST_REFERENCE",
        t_spinup    = Day(1),
        sim_days    = 10,
        fac_pert_t  = 2f0,
        fac_pert_q  = 0.2f0,
    ),

    # 1. No longwave radiation scheme         
    (;  name        = "FreeRun_default",
        lw_scheme   = nothing,
    ),

    # 2. OneBandLongwave default                  
    (;  name        = "OBLW_default"
    ),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME       = get(v, :name, "default")           # name of the reference data
SEED       = get(v, :seed, 1234)                # used seed

# Surface emissivity
EM_OCEAN    = get(v, :em_ocean, 0.98f0)
EM_LAND     = get(v, :em_land,  0.98f0)

# Model and target scheme
MODEL       = get(v, :model, PrimitiveWetModel)             # used model
LW_TARGET   = OneBandLongwave(SG;                           # used longwave radiation scheme
    transmissivity     = FriersonLongwaveTransmissivity(SG),
    radiative_transfer = OneBandLongwaveRadiativeTransfer(SG;
                            emissivity_ocean = EM_OCEAN,
                            emissivity_land  = EM_LAND),
)

MODEL      = get(v, :model, PrimitiveWetModel)  
LW_SCHEME  = get(v, :lw_scheme, LW_TARGET)        

# Sampling
T_SPINUP   = get(v, :t_spinup, Day(30))                     # spinup time in days
START_DATE = get(v, :start_date, DateTime(2001, 1, 1))      # sampling starting date
SIM_DAYS   = get(v, :sim_days, 2*365)                       # sampling time in days

# Perturbation
FAC_PERT_T = get(v, :fac_pert_t, 2f0)           # additive perturbation amplitude for temperature
FAC_PERT_Q = get(v, :fac_pert_q, 0.2f0)         # multiplicative perturbation amplitude for humidity (zeromin = true)



### Prepare generation
# Create output folder
DIR = prepare_out_dir(reference_dir(), NAME)

# Set seed for reproducability
Random.seed!(SEED)



### Generate the reference data set
ref = generate_reference(;
    spectral_grid   = SG,
    name            = NAME,
    dir             = DIR,
    seed            = SEED,

    model           = MODEL,
    lw_scheme       = LW_SCHEME,

    t_spinup        = T_SPINUP,
    start_date      = START_DATE,
    sim_days        = SIM_DAYS,

    fac_pert_T      = FAC_PERT_T,
    fac_pert_q      = FAC_PERT_Q,
)



### Create and store info.toml file
write_info(;
    dir = DIR,
    file = "info.toml",

    general = (;
        name    = NAME,
        created = now(),
        seed    = SEED,
        julia   = string(VERSION),
        sw_vers = string(pkgversion(SpeedyWeather)),
    ),

    grid = (;
        trunc   = TRUNC,
        nlayers = NLAYERS,
        grid_type   = string(nameof(SG.Grid)),
    ),

    model_type = (;
        model       = string(nameof(MODEL)),
        lw_scheme   = string(nameof(typeof(LW_SCHEME))),
    ),

    sampling = (;
        t_spinup      = string(T_SPINUP),
        start_date    = string(START_DATE),
        sim_days      = SIM_DAYS,
        steps_per_day = ref.steps_per_day,
    ),

    perturbation = (;
        fac_pert_t = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),
)
