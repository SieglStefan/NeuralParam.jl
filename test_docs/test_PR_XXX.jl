using SpeedyWeather
using CairoMakie

spectral_grid = SpectralGrid(trunc=42, nlayers=8)


initial_conditions = (; vordiv = RossbyHaurwitzWave(spectral_grid),
                        temp = JablonowskiTemperature(spectral_grid),
                        pres = PressureOnOrography(spectral_grid))

orography = NoOrography(spectral_grid)
time_stepping = Leapfrog(spectral_grid, Δt_at_T31=Minute(30))

forcing = nothing
drag = nothing

model = PrimitiveDryModel(spectral_grid; 
                            time_stepping, 
                            initial_conditions, 
                            orography, 
                            forcing, drag,
                            dynamics_only=true)     # XXX here is the difference

sim = initialize!(model)
run!(sim, period=Day(5))

vor = sim.variables.grid.vorticity[:, 8]
heatmap(vor, title="Relative vorticity [1/s], primitive Rossby-Haurwitz wave")