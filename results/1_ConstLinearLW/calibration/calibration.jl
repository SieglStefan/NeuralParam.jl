### XXX

using Revise
using NeuralParam
using SpeedyWeather
using Dates

TRUNC = 31
NLAYERS = 8

SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)

scaling = Scaling(NLAYERS)
scheme = ConstLinearLW(SG, user_scaling=scaling)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
output_path = joinpath(@__DIR__, "run_T$(TRUNC)_L$(NLAYERS)_$(timestamp)")
mkdir(output_path)

output_config = OutputConfig(output_path = output_path)
calibration_config = RunConfig(eta0 = 1f-2)

param, L, PN, GN = run_training(
    SG, 
    scheme; 
    run_config = calibration_config, 
    output_config, 
    test_mode=false)
