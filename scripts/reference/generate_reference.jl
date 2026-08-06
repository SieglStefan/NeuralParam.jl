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

    # 0. Default: no longwave radiation scheme
    (;),

    # 1. OneBandLongwave default
    (; name="OBLW_default", lw_scheme=OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG))),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME       = get(v, :name, "default")           # name of the reference data
SEED       = get(v, :seed, 1234)                # used seed

# Model and scheme
MODEL      = get(v, :model, PrimitiveWetModel)  # used model
LW_SCHEME  = get(v, :lw_scheme, nothing)        # used longwave radiation scheme

# Sampling
T_SPINUP   = get(v, :t_spinup, Day(30))                     # spinup time in days
START_DATE = get(v, :start_date, DateTime(2001, 1, 1))      # sampling starting date
SIM_DAYS   = get(v, :sim_days, 2*365)                       # sampling time in days
FULL_GAP   = get(v, :full_gap, 1)                           # store a full (restart) state every FULL_GAP days

# Perturbation
FAC_PERT_T = get(v, :fac_pert_t, 2f0)           # additive perturbation amplitude for temperature
FAC_PERT_Q = get(v, :fac_pert_q, 0.2f0)         # multiplicative perturbation amplitude for humidity (zeromin = true)



### Prepare generation
# Create output folder
DIR = prepare_out_dir(reference_dir(), NAME)



### Generate the reference data set
ref = generate_reference(
    SG;
    name        = NAME,
    dir         = DIR,
    seed        = SEED,
    model       = MODEL,
    lw_scheme   = LW_SCHEME,
    t_spinup    = T_SPINUP,
    start_date  = START_DATE,
    sim_days    = SIM_DAYS,
    full_gap    = FULL_GAP,
    fac_pert_T  = FAC_PERT_T,
    fac_pert_q  = FAC_PERT_Q,
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
        full_gap      = FULL_GAP,
        steps_per_day = ref.steps_per_day,
    ),

    perturbation = (;
        fac_pert_t = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),
)
