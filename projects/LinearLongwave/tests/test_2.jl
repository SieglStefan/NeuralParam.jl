using SpeedyWeather, CairoMakie, Random



spectral_grid = SpectralGrid(trunc=5, nlayers=1)

model_target1 = PrimitiveWetModel(; spectral_grid)

simulation_target1 = initialize!(model_target1)
run!(simulation_target1, period=Hour(12))


(; Δt, Δt_millisec) = model_target1.time_stepping
dt = 2Δt


for _ in 1:37
    SpeedyWeather.timestep!(simulation_target1.variables, dt, simulation_target1.model)
end

temp = simulation_target1.variables.grid.temperature[:, 1]
heatmap(temp, title="Temperature")