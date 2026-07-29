### ConstLinearLW Calibration
###
### Objective: Calibrate the parameters of a ConstLinearLW parameterization



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Dates



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
   


### Define possible variants for calibration
variants = [ 
    
    # Default: uses default parameters for calibration
    (;),                                                                                # 0
    
    # initialValue: uses different initial a and b for testing convergence of calibration scheme
    (; name = "iV1", a = fill(-0.1f0,     NLAYERS), b = fill(0.1f0,    NLAYERS)),       # 1
    (; name = "iV2", a = fill(-0.1f0,     NLAYERS), b = fill(10f0,     NLAYERS)),       # 2 
    (; name = "iV3", a = fill(-10f0,      NLAYERS), b = fill(0.1f0,    NLAYERS)),       # 3          
    (; name = "iV4", a = fill(-10f0,      NLAYERS), b = fill(10f0,     NLAYERS)),       # 4       
    (; name = "iV5", a = fill(-0.0001f0,  NLAYERS), b = fill(1f0,      NLAYERS)),       # 5                   
    (; name = "iV6", a = fill(-1f0,       NLAYERS), b = fill(0.0001f0, NLAYERS)),       # 6
    
    # dry: uses a PrimitiveDryModel instead of a PrimitiveWetModel
    (; name = "dry", model = PrimitiveDryModel, lw_scheme = OneBandLongwave(SG; transmissivity = ConstantLongwaveTransmissivity(SG))),          # 7
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME        = get(v, :name, "default")
SEED        = get(v, :seed, 1234)

# Model and scheme
MODEL       = get(v, :model, PrimitiveWetModel)
LW_SCHEME   = get(v, :lw_scheme, OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG)))

# Learning rate
ETA0        = get(v, :eta0, 1f-3)
ETA_DECAY   = get(v, :eta_decay, 0.9f0)

# Spinup and start date
T_SPINUP    = get(v, :t_spinup, Day(1))
START_DATE  = get(v, :start_date, DateTime(2000, 1, 1))

# Training steps parameter
N_IC        = get(v, :n_ic, 1)
N_TRAJ      = get(v, :n_traj, 10)
N_EPOCHS    = get(v, :n_epochs, 2)
N_BATCH     = get(v, :n_batch, 2)
N_STEPS_0   = get(v, :n_steps_0, 5)
N_STEPS_INC = get(v, :n_steps_inc, 2)
N_GAP       = get(v, :n_gap, 0)

# Perturbation
FAC_PERT_T  = get(v, :fac_pert_t, 2f0)
FAC_PERT_Q  = get(v, :fac_pert_q, 0.2f0)

# Initial parameters
a = get(v, :a, fill(-1f0, NLAYERS))
b = get(v, :b, fill(1f0, NLAYERS))
ps = (; a=a, b=b)



### Prepare calibration
# Create output folder
foldername = "CLLW_$(NAME)"
folderpath = prepare_out_dir(scheme_dir(), foldername)



### Calibration
# Define to be calibrated scheme
scheme = ConstLinearLW(Scaling(NLAYERS), ps)

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
    do_autodiff = false,
)

# Define output configuration
output_config = OutputConfig(output_path = folderpath, printing_traj=true)

# Run the calibration
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

    name        = NAME, 
    created     = now(), 
    seed        = SEED,
    julia       = string(VERSION), 
    sw          = string(pkgversion(SpeedyWeather)),

    trunc       = TRUNC, 
    nlayers     = NLAYERS,

    model       = nameof(MODEL), 
    lw_scheme   = nameof(typeof(LW_SCHEME)),

    eta0        = ETA0, 
    eta_decay   = ETA_DECAY,

    t_spinup    = string(T_SPINUP), 
    start_date  = string(START_DATE),

    n_ic        = N_IC, 
    n_traj      = N_TRAJ, 
    n_epochs    = N_EPOCHS,
    n_steps_0   = N_STEPS_0, 
    n_steps_inc = N_STEPS_INC, 
    n_gap       = N_GAP,

    fac_pert_t  = FAC_PERT_T, 
    fac_pert_q  = FAC_PERT_Q,

    scheme      = info_scheme(scheme_trained),
    final_loss  = L[end]
)