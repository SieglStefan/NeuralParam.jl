using SpeedyWeather, CairoMakie

using SpeedyWeather
spectral_grid = SpectralGrid(trunc=31, nlayers=4)

forcing = nothing
drag = nothing
model = BarotropicModel(spectral_grid; initial_conditions, planet=still_earth, forcing, drag)
simulation = initialize!(model)
run!(simulation, period=Day(20))