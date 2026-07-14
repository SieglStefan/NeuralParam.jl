### Plotting utility functions
###
### Plotting possibilities:
### - plot_loss:        
###         plots loss of a training run
### - plot_training:    
###         plots not only loss, but also parameter and gradient norm
### - plot_training_comp:
###         plots a loss comparison for several training runs
### - plot_comparison:
###         plots a compairson between two fields in respect to a given metric
### - plot_heatmap:
###         plots a single heatmap
### - plot_heatmaps:
###         plots 3 heatmaps with a shared colorbar
### - plot_histograms
###         plots histograms of fields for zscore validation








# Helper functions for creating a coastline
function field_to_lonlatmat(field)
    full = RingGrids.interpolate(RingGrids.full_grid_type(field.grid), field.grid.nlat_half, field)
    return RingGrids.get_lond(full), RingGrids.get_latd(full), Matrix(full)
end

shift_lon(lond, mat) = (lon = [l > 180 ? l - 360 : l for l in lond]; p = sortperm(lon); (lon[p], mat[p, :]))

finite_range(mats) = (v = filter(isfinite, vcat(vec.(mats)...)); (minimum(v), maximum(v)))


# Plot a single heatmap
function plot_heatmap(field; title = "Heatmap", coastlines = true, grid = false, kwargs...)
    lond, latd, mat = field_to_lonlatmat(field)
    fig = CairoMakie.Figure()

    if coastlines
        lon, mat = shift_lon(lond, mat)
        ax = GeoMakie.GeoAxis(fig[1, 1]; dest = "+proj=longlat", title = title, width = 500, height = 250,
                              xgridvisible = grid, ygridvisible = grid)
        hm = CairoMakie.heatmap!(ax, lon, latd, mat; kwargs...)
        CairoMakie.lines!(ax, GeoMakie.coastlines(); color = :black)
    else
        ax = CairoMakie.Axis(fig[1, 1]; title = title, width = 500, height = 250,
                             xgridvisible = grid, ygridvisible = grid)
        hm = CairoMakie.heatmap!(ax, lond, latd, mat; kwargs...)
    end

    CairoMakie.Colorbar(fig[1, 2], hm)
    CairoMakie.resize_to_layout!(fig)
    return fig
end


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





# XXX
function plot_histograms(data, titles; ncols=4, nbins=50, logx=false, suptitle="", size=(1500, 650))
    fig = CairoMakie.Figure(; size)
    for (k, (d, title)) in enumerate(zip(data, titles))
        d = filter(isfinite, d)
        logx && (d = log10.(filter(>(0), d)))
        ax = CairoMakie.Axis(fig[cld(k, ncols), mod1(k, ncols)];
                             title, titlesize = 12,
                             xlabel = logx ? "log₁₀" : "",
                             xticks = CairoMakie.LinearTicks(4),
                             xticklabelrotation = π/5, xticklabelsize = 10)
        CairoMakie.hist!(ax, d; bins = nbins)
        CairoMakie.text!(ax, 0.03, 0.97;
                         text = "μ=$(round(mean(d), sigdigits=4))\nσ=$(round(std(d), sigdigits=4))",
                         space = :relative, align = (:left, :top), fontsize = 9)
    end
    isempty(suptitle) || CairoMakie.Label(fig[0, :], suptitle; fontsize = 17, font = :bold)
    return fig
end