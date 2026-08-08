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

    # Test variant for quick testing                                    # 0: test
    (;  
        name        = "TEST_NLW",
        output_form = LinearOutput(),
        t_spinup    = Day(1),
        n_ic        = 2,
        n_traj      = 10,
        n_batch     = 2,
        n_steps_0   = 5,
        n_steps_inc = 2,
        n_gap       = 5,
    ),

    # Identity test - must provide zeros in parameters and loss         # 1: identity test
    (;  name = "TEST_IDENTITY",
        eta0 = 0f0,
        t_spinup    = Day(1),
        n_ic        = 2,
        n_traj      = 10,
        n_batch     = 2,
        n_steps_0   = 2,
        n_steps_inc = 1,
        n_gap       = 5,
    ),

    # Default NeuralLW
    (;  name = "NLLW_direct_default", output_form = DirectOutput()),    # 2: direct output
    (;  name = "NLLW_linear_default", output_form = LinearOutput()),    # 3: linear output
    (;  name = "NLLW_planck_default", output_form = PlanckOutput()),    # 4: planck output
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME        = get(v, :name, "default")
SEED        = get(v, :seed, 42)

# Scheme
INPUT_SPEC  = get(v, :inputs, INPUT_NLW_OBLW)
OUTPUT_FORM = get(v, :output_form, LinearOutput())
ARCH        = get(v, :arch, MLPConfig(n_hidden = 2, width = 32))
ZSCORE      = get(v, :zscore_name, "zscore_OBLW_default")

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
out_dir = startswith(NAME, "TEST") ? fresh_out_dir : prepare_out_dir
DIR = out_dir(scheme_dir(), NAME)

# Set seed for reproducability
Random.seed!(SEED)



### Training
# Define to be trained scheme
lw_train = NeuralLW(
    spectral_grid   = SG,
    arch_config     = ARCH,
    input_spec      = INPUT_SPEC,
    output_form     = OUTPUT_FORM,
    zscore_name     = ZSCORE;
    def_ocean_em    = EM_OCEAN,
    def_land_em     = EM_LAND
)

# Choose the same scheme as target if identity test
NAME == "TEST_IDENTITY" && (LW_TARGET = deepcopy(lw_train))


# Define loss config
loss_config = LossConfig(
    spectral_grid = SG, 
    zscore_name   = ZSCORE,
    weights       = WEIGHTS,
)

# Define run configuration
train_config = TrainConfig(
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
lw_trained = fetch(schedule(Task(() -> run_training(SG, lw_train, train_config), 1<<29)))


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
        model       = string(nameof(MODEL)),
        lw_target   = string(nameof(typeof(LW_TARGET))),
        emissivity_ocean = EM_OCEAN,
        emissivity_land  = EM_LAND,   
    ),

    scheme = (;
        inputs      = [string(k) for k in keys(INPUT_SPEC)],
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
        fac_pert_T = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),

    trained_scheme = (;
        scheme      = info_scheme(lw_trained),
    ),
)
