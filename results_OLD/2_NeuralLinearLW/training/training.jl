### XXX

using NeuralParam
using SpeedyWeather
using Dates

TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)


task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
N_IC_DEFAULT = parse(Int, get(ENV, "N_IC", "5"))



variants = [ (; ),              
    (; n_steps_0 = 5, n_steps_inc = 1, n_ic = 15),      # 19 steps
    (; n_steps_0 = 10, n_steps_inc = 2, n_ic = 10),     # 28 steps
    (; n_steps_0 = 20, n_steps_inc = 5, n_ic = 5),      # 40 steps

    (; width = 32, n_hidden = 1),
    (; width = 32, n_hidden = 2),
    (; width = 128, n_hidden = 1),
    (; width = 128, n_hidden = 2),

    (; model_type = PrimitiveDryModel),     # task 0 default
]


v = variants[task + 1]
n_ic = get(v, :n_ic, N_IC_DEFAULT)
width = get(v, :width, 32)
n_hidden = get(v, :n_hidden, 2)
n_steps_0 = get(v, :n_steps_0, 10)
n_steps_inc = get(v, :n_steps_inc, 5)

model_type = get(v, :model_type, PrimitiveWetModel)
transmissivity = model_type === PrimitiveDryModel ?
    ConstantLongwaveTransmissivity(SG) :
    FriersonLongwaveTransmissivity(SG)
target = OneBandLongwave(SG; transmissivity)



# --- Run-Ordner (gemeinsam für alle Tasks einer Array; lokal: Timestamp) ---
RUN = get(ENV, "RUN", "run_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))")
output_path = joinpath(@__DIR__, RUN, "task_$(task)")  

isdir(output_path) && error("Ordner existiert schon: $output_path — abgebrochen (nicht überschrieben).")
mkpath(output_path)



arch = MLPConfig(width = width, n_hidden = n_hidden)

scheme = NeuralLinearLW(SG, arch)
output_config = OutputConfig(output_path = output_path)

run_config = RunConfig(
    model_type = model_type,
    n_ic = n_ic,
    n_traj = 10,
    n_epochs = 5,
    n_steps_0 = n_steps_0,
    n_steps_inc = n_steps_inc,
    n_gap  = 25,
)

param, L, PN, GN = run_training(
    SG, 
    scheme, 
    target;
    run_config, 
    output_config, 
    test_mode=false)