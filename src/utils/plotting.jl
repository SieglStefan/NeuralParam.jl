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



# Plot timeseries of loss of a training run
function plot_loss(loss::AbstractVector; kwargs...)

    return Plots.plot(
        loss;
        xlabel = "Training step",
        ylabel = "Loss",
        yscale = :log10,
        title = "Loss over Training steps",
        lw = 2,
        kwargs...
    )
end

# Plot timeseries of loss of a training run from .csv file
function plot_loss(; path="", file="", kwargs...)
    
    # Read and extract data
    df = csv_read(;path, file)
    loss = df.loss
    
    return plot_loss(loss; kwargs...)
end



# Plot timeseries of loss, parameter- and gradient norm of a training run
function plot_training(
    loss::AbstractVector,
    pnorm::AbstractVector,
    gnorm::AbstractVector;
    loss_kwargs = (;),
    pnorm_kwargs = (;),
    gnorm_kwargs = (;),
    plot_kwargs = (;),
)

    # Loss plot
    p1 = Plots.plot(
        loss;
        ylabel = "Loss",
        yscale = :log10, 
        loss_kwargs...,
    )

    # Parameter norm plot
    p2 = Plots.plot(
        pnorm;
        ylabel = "Parameter norm",
        yscale = :log10, 
        pnorm_kwargs...,
    )

    # Gradient norm plot
    p3 = Plots.plot(
        gnorm;
        xlabel="Training step",
        ylabel="Gradient norm",
        yscale=:log10, 
        gnorm_kwargs...,
    )

    return Plots.plot(p1, p2, p3; layout=(3, 1), title="Training Values", plot_kwargs...)
end

# Plot timeseries of loss, parameter- and gradient norm of a training run from file
function plot_training(; path="", file="", kwargs...)
    
    # Read and extract data
    df = csv_read(;path, file)

    loss = df.loss
    pnorm = df.pnorm
    gnorm = df.gnorm
    
    return plot_training(loss, pnorm, gnorm; kwargs...)
end



# Plot timeseries of loss comparison of two training runs
function plot_training_comp(losses::AbstractVector{<:AbstractVector}; labels=nothing, kwargs...)

    # Create empty canvas
    p = Plots.plot(
        xlabel = "Training step",
        ylabel = "Loss",
        yscale = :log10,
        title = "Loss Comparison",
    )

    # Plot losses
    for (i, loss) in enumerate(losses)
        lab = isnothing(labels) ? "run $i" : labels[i]
        Plots.plot!(p, loss; label=lab, lw=2)
    end

    return Plots.plot(p; kwargs...)
end

# Plot timeseries of loss comparison of two training runs from file
function plot_training_comp(runs::AbstractVector{<:Tuple}; labels=nothing, kwargs...)

    # Read and extract data
    losses = [csv_read(; path=p, file=f).loss for (p,f) in runs]
    labs = isnothing(labels) ? [f for (_,f) in runs] : labels

    return plot_training_comp(losses; labels=labs, kwargs...)
end  



# Plot timeseries of two fields in respect to a given metric
function plot_comparison(
    target::NamedTuple,
    comp::NamedTuple;
    metric = rmse,
    Δt_sec,
    kwargs...,
)
    fields = keys(target)

    panels = map(enumerate(fields)) do (i, field)
        traj_t = target[field]
        traj_c = comp[field]
        t_days = (0:length(traj_t)-1) .* Δt_sec ./ (60 * 60 * 24)

        xlab = i == length(fields) ? "Time (days)" : ""
        Plots.plot(t_days, metric.(traj_t, traj_c);
                   ylabel = string(field), xlabel = xlab, legend = false)
    end

    return Plots.plot(
        panels...;
        layout = (length(panels), 1),
        plot_title = "Comparison ($(uppercase(string(metric))))",
        kwargs...,
    )
end



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


# Plot a multiple heatmaps with a shared colorbar
function plot_heatmaps(F_vec; titles = nothing, layout = :vertical, coastlines = true, grid = false, kwargs...)
    n      = length(F_vec)
    titles = isnothing(titles) ? ["Heatmap $i" for i in 1:n] : titles
    conv   = [field_to_lonlatmat(F) for F in F_vec]
    crange = finite_range([c[3] for c in conv])

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
    CairoMakie.resize_to_layout!(fig)
    return fig
end