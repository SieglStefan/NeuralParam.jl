### XXX


using NeuralParam
using SpeedyWeather
using Dates

TRUNC = 31
NLAYERS = 8

SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))   # 0..4 (lokal Default 0)

# 5 verschiedene Start-a,b (alle a<0)
inits = [
    (; a = fill(-0.01f0, NLAYERS), b = fill(0.01f0, NLAYERS)),
    (; a = fill(-0.1f0, NLAYERS), b = fill(0.1f0,   NLAYERS)),
    (; a = fill(-1f0,   NLAYERS), b = fill(1f0,   NLAYERS)),
    (; a = fill(-10f0,   NLAYERS), b = fill(10f0, NLAYERS)),
    (; a = fill(-100f0,   NLAYERS), b = fill(100f0,   NLAYERS)),
]
ps0 = inits[task + 1]                                     # Julia 1-basiert → +1



scaling = Scaling(NLAYERS)
scheme = ConstLinearLW(scaling, ps0)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")




job_id = get(ENV, "SLURM_ARRAY_JOB_ID", "local")
output_path = joinpath(
    @__DIR__,
    "init_test",
    "job_$(job_id)",
    "init_$(task)",
)






mkpath(output_path)

output_config = OutputConfig(output_path = output_path)
calibration_config = RunConfig(eta0 = 1f-2)

param, L, PN, GN = run_training(
    SG, 
    scheme; 
    run_config = calibration_config, 
    output_config, 
    test_mode=false)
