### ConstLinearLW Calibration
###
### XXX



# Load packages
using NeuralParam
using SpeedyWeather



### Parameters and controls
# Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
   

# Define possible variants for calibration
variants = [ 
    
    # default: uses default parameters for calibration
    (;),                                                                            # 0
    
    # initialValue: uses different initial a and b for testing convergence of calibration scheme
    (; a = fill(-0.1f0,     NLAYERS), b = fill(0.1f0,    NLAYERS)),                 # 1
    (; a = fill(-0.1f0,     NLAYERS), b = fill(10f0,     NLAYERS), n_epochs=20),    # 2 (higher n_epochs because small convergence)            
    (; a = fill(-10f0,      NLAYERS), b = fill(0.1f0,    NLAYERS), n_epochs=20),    # 3 (-//-)          
    (; a = fill(-10f0,      NLAYERS), b = fill(10f0,     NLAYERS), n_epochs=20),    # 4 (-//-)        
    (; a = fill(-0.0001f0,  NLAYERS), b = fill(1f0,      NLAYERS)),                 # 5                   
    (; a = fill(-1f0,       NLAYERS), b = fill(0.0001f0, NLAYERS)),                 # 6
    
    # dry: uses a PrimitiveDryModel instead of a PrimitiveWetModel
    (; model_type = PrimitiveDryModel),                                             # 7
]


# Get environment variables
RUN_NAME = get(ENV, "RUN", "run")
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
N_IC_DEFAULT = parse(Int, get(ENV, "N_IC", "5")) 


# Choose a specific task for a job
v = variants[task + 1]

# Extract parameters of task
a = get(v, :a, fill(-1f0, NLAYERS))
b = get(v, :b, fill(1f0, NLAYERS))
ps = (; a=a, b=b)

n_ic = get(v, :n_ic, N_IC_DEFAULT)
n_epochs = get(v, :n_epochs, 10)
model_type = get(v, :model_type, PrimitiveWetModel)



# Create output folder
output_path = prepare_out_dir(joinpath(@__DIR__, RUN_NAME), "task_$(task)")



# Define target model
transmissivity = model_type === PrimitiveDryModel ?
    ConstantLongwaveTransmissivity(SG) :
    FriersonLongwaveTransmissivity(SG)
target = OneBandLongwave(SG; transmissivity)



### Calibration
# Define to be calibrated scheme
scheme = ConstLinearLW(Scaling(NLAYERS), ps)

# Define run configuration
run_config = RunConfig(
    model_type = model_type,
    eta0 = 1f-2, 
    n_ic = n_ic,
    n_traj = 10,
    n_epochs = n_epochs,
    n_steps_0 = 10,
    n_steps_inc = 2,
    n_gap = 10
)

# Define output configuration
output_config = OutputConfig(output_path = output_path)

# Run the calibration
param, L, PN, GN = run_training(
    SG,
    scheme,
    target; 
    run_config, 
    output_config, 
    test_mode = false
)