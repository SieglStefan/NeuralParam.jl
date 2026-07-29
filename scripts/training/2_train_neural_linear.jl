### NeuralLinearLW Training
###
### Objective: Train the NN parameters of a NeuralLinearLW parameterization



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Dates, Random



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
   


### Define possible variants for training
variants = [ 
    
    # Default: uses default parameters for training 
    (;),                                          
    
    # Longer trajectories
    (; name = "long_traj", n_traj = 200),
    
    # Hyperparameters of NN
    (; name = "W32_H1", width = 32, n_hidden = 1),
    (; name = "W32_H2", width = 32, n_hidden = 2),
    (; name = "W128_H1", width = 128, n_hidden = 1),
    (; name = "W128_H2", width = 128, n_hidden = 2),

]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME        = get(v, :name, "default")
SEED        = get(v, :seed, 42)

# Model and scheme
MODEL       = get(v, :model, PrimitiveWetModel)
LW_SCHEME   = get(v, :lw_scheme, OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG)))

# Learning rate
ETA0        = get(v, :eta0, 1f-2)
ETA_DECAY   = get(v, :eta_decay, 0.8f0)

# Spinup and start date
T_SPINUP    = get(v, :t_spinup, Day(30))
START_DATE  = get(v, :start_date, DateTime(2000, 1, 1))

# Training steps parameter
N_IC        = get(v, :n_ic, 5)
N_TRAJ      = get(v, :n_traj, 100)
N_EPOCHS    = get(v, :n_epochs, 1)
N_BATCH     = get(v, :n_batch, 4)
N_STEPS_0   = get(v, :n_steps_0, 5)
N_STEPS_INC = get(v, :n_steps_inc, 2)
N_GAP       = get(v, :n_gap, 50)

# Perturbation
FAC_PERT_T  = get(v, :fac_pert_t, 2f0)
FAC_PERT_Q  = get(v, :fac_pert_q, 0.2f0)

# NN Architecture
WIDTH       = get(v, :width, 32)
N_HIDDEN    = get(v, :n_hidden, 1)

# Statistics
ZCSORE      = get(v, :zscore, "zscore_llw_default")  
SCALING     = get(v, :scaling, "scaling_llw_default")



### Prepare training
# Create output folder
foldername = "NLLW_$(NAME)"
folderpath = prepare_out_dir(scheme_dir(), foldername)

# Set seed for reproducability
Random.seed!(SEED)


# Define NN architecture
arch = MLPConfig(width = WIDTH, n_hidden = N_HIDDEN)

# Define to be trained scheme
scheme = NeuralLinearLW(SG, arch; zscore_folder=ZCSORE, scaling_folder=SCALING)

# Warmstart the scheme with a default ConstLinearLW parameters
warmstart!(scheme)


# Define run configuration
run_config = RunConfig(
    seed        = SEED,
    model       = MODEL,
    lw_scheme   = LW_SCHEME,
    eta0        = ETA0,
    eta_decay   = ETA_DECAY, 
    t_spinup    = T_SPINUP,
    start_date  = START_DATE,
    n_ic        = N_IC,
    n_traj      = N_TRAJ,
    n_epochs    = N_EPOCHS,
    n_batch     = N_BATCH,
    n_steps_0   = N_STEPS_0,
    n_steps_inc = N_STEPS_INC,
    n_gap       = N_GAP,
    fac_pert_T  = FAC_PERT_T,
    fac_pert_q  = FAC_PERT_Q,
    do_autodiff = true,
)

# Define output configuration
output_config = OutputConfig(output_path = folderpath)


# Run the training
scheme_trained, L, PN, GN = run_training(
    SG,
    scheme; 
    run_config, 
    output_config, 
)


# Create and store info.toml file
write_info(; 
    path = folderpath, 
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
        model       = nameof(MODEL), 
        lw_scheme   = nameof(typeof(LW_SCHEME)),
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
        n_epochs    = N_EPOCHS,
        n_batch     = N_BATCH,
        n_steps_0   = N_STEPS_0, 
        n_steps_inc = N_STEPS_INC, 
        n_gap       = N_GAP,
    ),

    perturbation = (;
        fac_pert_t = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),

    stats = (;
        zscore      = ZCSORE,
        scaling     = SCALING,
        warmstart   = true,
    ),

    final = (;
        scheme      = info_scheme(scheme_trained),
        final_loss  = L[end]
    ),
)