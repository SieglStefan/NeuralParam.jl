using SpeedyWeather
using CairoMakie
using GeoMakie
using Dates

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

spectral_grid = SpectralGrid(trunc = 31, nlayers = 1)

model = PrimitiveWetModel(; spectral_grid)
sim = initialize!(model)

run!(sim, period = Day(14))

T = sim.variables.grid.temperature[:, 1]

# ------------------------------------------------------------
# Plot native SpeedyWeather/RingGrids heatmap + coastlines
# ------------------------------------------------------------

fig, ax, hm = CairoMakie.heatmap(
    T;
    axis = (
        title = "Temperature after 14 day spinup",
        xlabel = "Longitude",
        ylabel = "Latitude",
    ),
)

lines!(
    ax,
    GeoMakie.coastlines(10);
    color = :black,
    linewidth = 0.8,
)

Colorbar(fig[1, 2], hm; label = "Temperature [K]")

display(fig)