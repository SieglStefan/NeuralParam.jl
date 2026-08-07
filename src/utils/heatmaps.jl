### Heatmap utilities
###
### Functions for plotting heatmaps, primarly used in evaluation/rollouts.jl
###     - plot_heatmap:         plots a single heatmap of a field
###     - plot_heatmaps:        plots multiple heatmaps of fields with shared colorbar



# Helper functions for creating a coastline overlay
function field_to_lonlatmat(field)
    full = RingGrids.interpolate(RingGrids.full_grid_type(field.grid), field.grid.nlat_half, field)
    return RingGrids.get_lond(full), RingGrids.get_latd(full), Matrix(full)
end

shift_lon(lond, mat) = (lon = [l > 180 ? l - 360 : l for l in lond]; p = sortperm(lon); (lon[p], mat[p, :]))


# Finite values of several fields/matrices, flattened
_finite_values(mats) = filter(isfinite, vcat(vec.(mats)...))

# Helper functions for creating a color range for heatmaps
finite_range(mats) = (v = _finite_values(mats); (minimum(v), maximum(v)))

# Symmetric color range around zero, for difference maps
sym_range(mats) = (m = maximum(abs, _finite_values(mats)); (-m, m))



# Plot multiple heatmaps with shared colorbar
function plot_heatmaps(F_vec; titles = nothing, layout = :vertical, coastlines = true, grid = false, suptitle = "", colorrange=nothing, kwargs...)
    n      = length(F_vec)
    titles = isnothing(titles) ? ["Heatmap $i" for i in 1:n] : titles
    conv   = [field_to_lonlatmat(F) for F in F_vec]
    crange = isnothing(colorrange) ? finite_range([c[3] for c in conv]) : colorrange

    fig = CairoMakie.Figure()
    hm  = nothing
    for (i, (lond, latd, mat)) in enumerate(conv)
        pos = layout == :vertical ? fig[i, 1] : fig[1, i]
        if coastlines
            lon, mat = shift_lon(lond, mat)
            ax = GeoMakie.GeoAxis(pos; dest = "+proj=longlat", title = titles[i], width = 500, height = 250,
                                  xgridvisible = grid, ygridvisible = grid)
            hm = CairoMakie.heatmap!(ax, lon, latd, mat; colorrange = crange, kwargs...)
            CairoMakie.lines!(ax, GeoMakie.coastlines(); color = :black)
        else
            ax = CairoMakie.Axis(pos; title = titles[i], width = 500, height = 250,
                                 xgridvisible = grid, ygridvisible = grid)
            hm = CairoMakie.heatmap!(ax, lond, latd, mat; colorrange = crange, kwargs...)
        end
    end

    layout == :vertical ? CairoMakie.Colorbar(fig[:, 2], hm) : CairoMakie.Colorbar(fig[1, n+1], hm)

    isempty(suptitle) || CairoMakie.Label(fig[0, :], suptitle; fontsize = 18, font = :bold)
    CairoMakie.resize_to_layout!(fig)
    return fig
end