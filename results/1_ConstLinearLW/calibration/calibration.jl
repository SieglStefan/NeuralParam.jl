### XXX


using NeuralParam
using SpeedyWeather
using Dates


TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
N_IC_DEFAULT = parse(Int, get(ENV, "N_IC", "5"))     # von launch.sh, sonst 5



task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))

variants = [ 
    
    (; a=fill(-1f0,NLAYERS),   b=fill(1f0,NLAYERS)),     # task 0 default
    
    (; a = fill(-0.1f0, NLAYERS), b = fill(0.1f0,   NLAYERS)),                  # task 1  ┐
    (; a = fill(-0.1f0, NLAYERS), b = fill(10f0,    NLAYERS)),                  # task 2  │ nur für Multi
    (; a = fill(-10f0,  NLAYERS), b = fill(0.1f0,   NLAYERS)),                  # task 3  │ (Init-Test)
    (; a = fill(-10f0,  NLAYERS), b = fill(10f0,    NLAYERS)),      # task 4  ┘

    (; a = fill(-0.0001f0, NLAYERS), b = fill(1f0, NLAYERS)),                # task 1  ┐
    (; a = fill(-1f0, NLAYERS), b = fill(0.0001f0, NLAYERS)),                  # task 2  │ nur für Multi
]


v = variants[task + 1]
ps = (; a=v.a, b=v.b)
n_ic = get(v, :n_ic, N_IC_DEFAULT)


# --- Run-Ordner (gemeinsam für alle Tasks einer Array; lokal: Timestamp) ---
RUN = get(ENV, "RUN", "run_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))")
output_path = joinpath(@__DIR__, RUN, "task_$(task)")  

isdir(output_path) && error("Ordner existiert schon: $output_path — abgebrochen (nicht überschrieben).")
mkpath(output_path)


# --- Kalibrieren ---
scheme = ConstLinearLW(Scaling(NLAYERS), ps)
run_config    = RunConfig(
    eta0 = 1f-2, 
    n_ic = n_ic,
    n_traj = 10,
    n_epochs = 5,
    n_steps_0 = 10,
    n_steps_inc = 2,
    n_gap = 10)

output_config = OutputConfig(output_path = output_path)

param, L, PN, GN = run_training(SG, scheme; run_config, output_config, test_mode = false)