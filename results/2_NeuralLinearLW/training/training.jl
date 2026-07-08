### XXX

using Revise
using NeuralParam
using SpeedyWeather
using Dates

TRUNC = 31
NLAYERS = 8


SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
arch = MLPConfig()

lw_radiation_target = OneBandLongwave(SG)

scheme = NeuralABRLW(SG, arch)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
output_path = joinpath(@__DIR__, "run_T$(TRUNC)_L$(NLAYERS)_$(timestamp)")
mkdir(output_path)

output_config = OutputConfig(output_path = output_path)

run_config = RunConfig()

param, L, PN, GN = run_training(
    SG, 
    scheme; 
    lw_radiation_target,
    run_config, 
    output_config, 
    test_mode=true)