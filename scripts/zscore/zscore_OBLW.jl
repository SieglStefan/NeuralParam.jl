### Script for calculating zscore stats for a OneBandLongwave (LW) target scheme
###
### Objective: Calculate statistics for:
### - input variables:    T, q, p, lat, lf, sst, lst, Ts
### - direct outputs:     dT, olw, slwd
### - linear outputs:     a, b (per layer), c, d, e (olw), f, g (slwd)
###
### Additionally:
###     - stores meta data of creation of stats at info.toml
###     - stores a subsample of the raw regression data for later plotting
###     - stores histograms of normalized variables and regression scatter plots



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



### Define parameters for sampling
# General
NAME        = "zscore_OBLW_default"     # name of statistics
SEED        = 1234                      # seed used

# Sampling
T_SPINUP    = Day(30)                   # spinup time
START_DATE  = DateTime(2000, 1, 1)      # start date of simulation
N_IC        = 1                         # number of initial conditions
SIM_TIME    = 365                       # sampling time in days
SAMPLE_GAP  = 3.65                      # days between sampling

# Perturbation
FAC_PERT_T  = 2f0                       # temperature perturbation amplitude
FAC_PERT_Q  = 0.2f0                     # humidity perturbation amplitude (multiplicative)

# Output forms: one stats group is fitted per form, keyed by output_group(form)
OUTPUT_FORMS = (DirectOutput(), LinearOutput(), PlanckOutput())

# Surface emissivity
EM_OCEAN    = 0.98f0
EM_LAND     = 0.98f0

# Model and scheme
MODEL       = PrimitiveWetModel
LW_SCHEME   = OneBandLongwave(SG;
    transmissivity     = FriersonLongwaveTransmissivity(SG),
    radiative_transfer = OneBandLongwaveRadiativeTransfer(SG;
                            emissivity_ocean = EM_OCEAN,
                            emissivity_land  = EM_LAND),
)



### Prepare generation
# Create output folder
DIR = prepare_out_dir(stats_dir(), NAME)

# Set seed for reproducability
Random.seed!(SEED)



### Generate zscore statistics
zs = generate_zscore(;
    spectral_grid   = SG,
    name            = NAME,
    dir             = DIR,
    seed            = SEED,

    model           = MODEL,
    lw_scheme       = LW_SCHEME,

    output_forms    = OUTPUT_FORMS,

    t_spinup        = T_SPINUP,
    start_date      = START_DATE,
    n_ic            = N_IC,
    sim_time        = SIM_TIME,
    sample_gap      = SAMPLE_GAP,

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

    io = (;
        inputs  = ["T", "q", "p", "lat", "lf", "sst", "lst", "Ts"],
        groups  = [string(output_group(f)) for f in OUTPUT_FORMS],
        keys    = [string(nameof(typeof(f))) * ": " * join(string.(output_keys(f)), ", ")
                   for f in OUTPUT_FORMS],
        forms   = ["affine: dT = a*P(T) + b", "olw = c + d*P(T[nlayers÷2]) + e*P(Ts)",
                   "slwd = f + g*P(T[nlayers])", "P = identity (linear) or x^4 (planck)"],
    ),

    grid = (;
        trunc     = TRUNC,
        nlayers   = NLAYERS,
        grid_type = string(nameof(SG.Grid)),
    ),

    model_type = (;
        model            = string(nameof(MODEL)),
        lw_scheme        = string(nameof(typeof(LW_SCHEME))),
        transmissivity   = string(nameof(typeof(LW_SCHEME.transmissivity))),
        emissivity_ocean = LW_SCHEME.radiative_transfer.emissivity_ocean,
        emissivity_land  = LW_SCHEME.radiative_transfer.emissivity_land,
    ),

    sampling = (;
        t_spinup   = string(T_SPINUP),
        start_date = string(START_DATE),
        n_ic       = N_IC,
        sim_time   = SIM_TIME,
        sample_gap = SAMPLE_GAP,
    ),

    perturbation = (;
        fac_pert_t = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),
)
