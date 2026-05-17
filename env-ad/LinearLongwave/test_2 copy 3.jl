using SpeedyWeather, CairoMakie, Enzyme


spectral_grid = SpectralGrid(trunc=31, nlayers=4)
model = PrimitiveWetModel(spectral_grid)
sim = initialize!(model)


run!(sim, period=Hour(1))




T_grid = deepcopy(sim.variables.grid.temperature)


T_grid .= 273.


sim2 = initialize!(model)
set!(sim2, temperature=T_grid)

k = 4

initialize!(sim2)

display(heatmap(sim2.variables.grid.temperature[:,k].-273, title="Day 0"))

run!(sim2, period=Day(1))

display(heatmap(sim2.variables.grid.temperature[:,k].-273, title="Day 1"))