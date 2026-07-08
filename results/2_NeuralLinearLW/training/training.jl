### XXX

using NeuralParam
using SpeedyWeather
using Dates

TRUNC = 31
NLAYERS = 8


SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
arch = MLPConfig()

lw_radiation_target = OneBandLongwave(SG)

scheme = NeuralLinearLW(SG, arch)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
output_path = joinpath(@__DIR__, "run_T$(TRUNC)_L$(NLAYERS)_$(timestamp)")
mkdir(output_path)

output_config = OutputConfig(output_path = output_path)

run_config = RunConfig(
    seed = 42,
    n_traj = 20,
    n_epochs = 5,
    n_steps_0 = 20,
    n_steps_inc = 10,
)

param, L, PN, GN = run_training(
    SG, 
    scheme; 
    lw_radiation_target,
    run_config, 
    output_config, 
    test_mode=false)