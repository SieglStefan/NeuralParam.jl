using SpeedyWeather
using RingGrids
using CairoMakie
using GeoMakie
using Dates

# ------------------------------------------------------------
# Helper: interpolate a RingGrids field onto a regular lon-lat grid
# ------------------------------------------------------------
function regularize_field(F; dlon = 1.0)
    lon, lat = RingGrids.get_londlatds(F.grid)
    vals = F.data

    lat_rings = sort(unique(lat))
    lon_grid = collect(-180.0:dlon:180.0)

    Z = Matrix{Float32}(undef, length(lat_rings), length(lon_grid))

    for (i, ϕ) in enumerate(lat_rings)
        idx = findall(==(ϕ), lat)

        λ = lon[idx]
        v = vals[idx]

        p = sortperm(λ)
        λ = λ[p]
        v = v[p]

        # periodic extension in longitude
        λ_ext = vcat(λ[end] - 360.0, λ, λ[1] + 360.0)
        v_ext = vcat(v[end], v, v[1])

        for (j, λq) in enumerate(lon_grid)
            k = searchsortedlast(λ_ext, λq)
            k = clamp(k, 1, length(λ_ext) - 1)

            λ1, λ2 = λ_ext[k], λ_ext[k + 1]
            v1, v2 = v_ext[k], v_ext[k + 1]

            α = (λq - λ1) / (λ2 - λ1)
            Z[i, j] = (1 - α) * v1 + α * v2
        end
    end

    return lon_grid, lat_rings, Z
end

# ------------------------------------------------------------
# Create a small SpeedyWeather simulation
# ------------------------------------------------------------
spectral_grid = SpectralGrid(trunc = 31, nlayers = 1)

model = PrimitiveWetModel(; spectral_grid)
sim = initialize!(model)

run!(sim, period = Day(1))

# Extract one temperature field
T = sim.variables.grid.temperature[:, 1]

# Interpolate to regular lon-lat grid
lon_grid, lat_grid, Z = regularize_field(T; dlon = 1.0)

# ------------------------------------------------------------
# Plot normal heatmap + coastlines
# ------------------------------------------------------------
fig = Figure(size = (1100, 550))

ax = GeoAxis(
    fig[1, 1];
    dest = "+proj=eqc",
    title = "Temperature with coastlines",
)

hm = heatmap!(
    ax,
    lon_grid,
    lat_grid,
    permutedims(Z);   # Makie wants (x, y, z) with z transposed here
    colormap = :viridis,
)

lines!(
    ax,
    GeoMakie.coastlines(10);   # 10 = fine coastlines
    color = :black,
    linewidth = 0.8,
)

Colorbar(fig[1, 2], hm; label = "Temperature [K]")

display(fig)