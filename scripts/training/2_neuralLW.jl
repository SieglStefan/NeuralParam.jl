### NeuralLW Training
###
### Train the neural network of a NeuralLW scheme against a OneBandLongwave (OBLW) target



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Dates, Random



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc = TRUNC, nlayers = NLAYERS)



### Define possible variants for training
variants = [

    # 0: AD smoke test — few updates, short gradient trajectories and a tiny network,
    #    just enough to see the loss move and confirm the adjoint runs through Lux
    (;  name        = "NLLW_test_ad",
        output_form = LinearOutput(),
        arch        = MLPConfig(n_hidden = 1, width = 16),
        t_spinup    = Day(10),
        n_ic        = 2,
        n_traj      = 10,
        n_batch     = 2,
        n_steps_0   = 5,
        n_steps_inc = 2,
        n_gap       = 20,
    ),

    # Default NeuralLW
    (;  name = "NLLW_default_linear", output_form = LinearOutput()),    # 1: linear output
    (;  name = "NLLW_default_direct", output_form = DirectOutput()),    # 2: direct output
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME        = get(v, :name, "default")
SEED        = get(v, :seed, 42)

# Scheme
INPUTS_SPEC = get(v, :inputs, IN_NLW_OBLW)
OUTPUT_FORM = get(v, :output_form, LinearOutput())
ARCH        = get(v, :arch, MLPConfig(n_hidden = 2, width = 32))
ZSCORE      = get(v, :zscore_name, "zscore_oblw_layer_4")

# Surface emissivity
EM_OCEAN    = get(v, :em_ocean, 0.98f0)
EM_LAND     = get(v, :em_land,  0.98f0)

# Model and target scheme
MODEL       = get(v, :model, PrimitiveWetModel)
LW_TARGET   = OneBandLongwave(SG;
    transmissivity     = FriersonLongwaveTransmissivity(SG),
    radiative_transfer = OneBandLongwaveRadiativeTransfer(SG;
                            emissivity_ocean = EM_OCEAN,
                            emissivity_land  = EM_LAND),
)

# Loss weighting
WEIGHTS     = get(v, :weights, (; T = 1f0, olw = 1f0, slwd = 1f0))

# Learning rate
ETA0        = get(v, :eta0, 1f-3)
ETA_DECAY   = get(v, :eta_decay, 0.7f0)

# Spinup and start date
T_SPINUP    = get(v, :t_spinup, Day(30))
START_DATE  = get(v, :start_date, DateTime(2000, 1, 1))

# Training steps parameter
N_IC        = get(v, :n_ic, 5)
N_TRAJ      = get(v, :n_traj, 80)
N_BATCH     = get(v, :n_batch, 4)
N_STEPS_0   = get(v, :n_steps_0, 5)
N_STEPS_INC = get(v, :n_steps_inc, 2)
N_GAP       = get(v, :n_gap, 50)

# Perturbation
FAC_PERT_T  = get(v, :fac_pert_T, 2f0)
FAC_PERT_Q  = get(v, :fac_pert_q, 0.2f0)



### Prepare training
# Create output folder
DIR = prepare_out_dir(scheme_dir(), NAME)

# Set seed for reproducability
Random.seed!(SEED)



### Training
# Define to be trained scheme
scheme = NeuralLW(
    SG,
    ARCH,
    INPUTS_SPEC,
    OUTPUT_FORM,
    ZSCORE;
    def_ocean_em = EM_OCEAN,
    def_land_em = EM_LAND
)

# Define loss config
loss_config = LossConfig(
    SG,
    ZSCORE;
    weights = WEIGHTS,
)

# Define run configuration
run_config = RunConfig(
    seed        = SEED,
    name        = NAME,
    dir         = DIR,
    model       = MODEL,
    lw_target   = LW_TARGET,
    eta0        = ETA0,
    eta_decay   = ETA_DECAY,
    loss_config = loss_config,
    t_spinup    = T_SPINUP,
    start_date  = START_DATE,
    n_ic        = N_IC,
    n_traj      = N_TRAJ,
    n_batch     = N_BATCH,
    n_steps_0   = N_STEPS_0,
    n_steps_inc = N_STEPS_INC,
    n_gap       = N_GAP,
    fac_pert_T  = FAC_PERT_T,
    fac_pert_q  = FAC_PERT_Q,
    do_autodiff = true,
)


# Run the training
lw_trained = fetch(schedule(Task(() -> run_training(SG, scheme, run_config), 1<<29)))


# Create and store info.toml file
write_info(;
    dir = DIR,
    file = "info.toml",

    general = (;
        name        = NAME,
        created     = now(),
        seed        = SEED,
        julia       = string(VERSION),
        sw_vers     = string(pkgversion(SpeedyWeather)),
    ),

    grid = (;
        trunc       = TRUNC,
        nlayers     = NLAYERS,
        grid_type   = string(nameof(SG.Grid)),
    ),

    model_type = (;
        model            = string(nameof(MODEL)),
        lw_target        = string(nameof(typeof(LW_TARGET))),
        transmissivity   = string(nameof(typeof(LW_TARGET.transmissivity))),
        emissivity_ocean = LW_TARGET.radiative_transfer.emissivity_ocean,
        emissivity_land  = LW_TARGET.radiative_transfer.emissivity_land,
    ),

    scheme = (;
        inputs      = [string(k) for k in keys(INPUTS_SPEC)],
        output_form = string(nameof(typeof(OUTPUT_FORM))),
        n_hidden    = ARCH.n_hidden,
        width       = ARCH.width,
        act         = string(ARCH.act),
    ),

    loss = (;
        weights = WEIGHTS,
        zscore  = ZSCORE,
    ),

    learning_rate = (;
        eta0        = ETA0,
        eta_decay   = ETA_DECAY,
    ),

    training_time = (;
        t_spinup    = string(T_SPINUP),
        start_date  = string(START_DATE),
    ),

    steps = (;
        n_ic        = N_IC,
        n_traj      = N_TRAJ,
        n_batch     = N_BATCH,
        n_steps_0   = N_STEPS_0,
        n_steps_inc = N_STEPS_INC,
        n_gap       = N_GAP,
    ),

    perturbation = (;
        fac_pert_t = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),

    trained_scheme = (;
        scheme      = info_scheme(lw_trained),
    ),
)
