using SpeedyWeather, CairoMakie, Enzyme


spectral_grid = SpectralGrid(trunc=31, nlayers=4)
model = PrimitiveWetModel(spectral_grid)
sim = initialize!(model)


run!(sim, period=Hour(1))



T_grid = make_zero(sim.variables.grid.temperature)
T_grid .+= 300.


# grid -> spectral
T_spec = sim.variables.prognostic.temperature[:,:,2]
transform!(T_spec, T_grid, model.spectral_transform)


#run!(sim, period=Second(0))

T_grid_new = make_zero(sim.variables.grid.temperature)


# spectral -> grid, damit diagnostics aktuell sind
transform!(T_grid_new, T_spec, model.spectral_transform)

T_grid_new


run!(sim, period=Second(0))

heatmap(T_grid[:, 1])