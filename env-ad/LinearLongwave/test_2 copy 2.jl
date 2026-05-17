using SpeedyWeather, CairoMakie, Enzyme


spectral_grid = SpectralGrid(trunc=31, nlayers=4)
model = PrimitiveWetModel(spectral_grid)
sim = initialize!(model)




run!(sim, period=Hour(1))




T_grid = deepcopy(sim.variables.grid.temperature)


noise = 10. .* (randn(Float32, size(T_grid)))


size(T_grid)


T_grid .+= noise


# grid -> spectral
T_spec = sim.variables.prognostic.temperature[:,:,2]
transform!(T_spec, T_grid, model.spectral_transform)


sim2 = initialize!(model)

set!(sim2, temperature=T_spec)

k = 4

initialize!(sim2)

display(heatmap(sim2.variables.grid.temperature[:,k].-273, title="Day 0"))

run!(sim2, period=Day(1))

display(heatmap(sim2.variables.grid.temperature[:,k].-273, title="Day 1"))

run!(sim2, period=Day(12))

display(heatmap(sim2.variables.grid.temperature[:,k].-273, title="Day 12"))