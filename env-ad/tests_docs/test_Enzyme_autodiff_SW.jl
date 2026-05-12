# First run ~ 1 hour

using SpeedyWeather, Enzyme, Test

spectral_grid = SpectralGrid(trunc=5, nlayers=1)
model = PrimitiveWetModel(; spectral_grid)
simulation = initialize!(model)
initialize!(simulation)
run!(simulation, period = Hour(6)) # spin-up the model a bit



(; variables, model) = simulation
(; Δt, Δt_millisec) = model.time_stepping
dt = 2Δt

dvars = make_zero(variables)
dvars.prognostic.vorticity .= 1 + im
dvars.prognostic.divergence .= 1 + im
dvars.prognostic.temperature .= 1 + im
dvars.prognostic.pressure .= 1 + im
dvars.prognostic.humidity .= 1 + im

dmodel = make_zero(model) # here, we'll accumulate all parameter derivatives


autodiff(Reverse, SpeedyWeather.timestep!, Const, Duplicated(variables, dvars), Const(dt), Duplicated(model, dmodel))

@test sum(abs, dvars.prognostic.vorticity) != 0
@test sum(abs, dvars.prognostic.divergence) != 0
@test sum(abs, dvars.prognostic.temperature) != 0
@test sum(abs, dvars.prognostic.pressure) != 0
@test sum(abs, dvars.prognostic.humidity) != 0

@show dvars.prognostic.vorticity

@show dmodel.atmosphere
@show dmodel.planet