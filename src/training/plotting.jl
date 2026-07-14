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
        legend = false, 
        loss_kwargs...,
    )

    # Parameter norm plot
    p2 = Plots.plot(
        pnorm;
        ylabel = "Parameter norm",
        yscale = :log10,
        legend = false,  
        pnorm_kwargs...,
    )

    # Gradient norm plot
    p3 = Plots.plot(
        gnorm;
        xlabel="Training step",
        ylabel="Gradient norm",
        yscale=:log10,
        legend = false,  
        gnorm_kwargs...,
    )

    defaults = (; size = (600, 900), left_margin = 8Plots.mm)
    merged = merge(defaults, plot_kwargs)

    return Plots.plot(p1, p2, p3; layout=(3, 1), merged...)
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



# Plot per-component metrics from a training .csv:
function plot_metrics(; path="", file="", fraction=1.0, kwargs...)
    df = csv_read(; path, file)

    # keep only the last `fraction` of rows (at least 1), preserving original step numbers
    n  = nrow(df)
    k  = max(1, round(Int, n * clamp(fraction, 0, 1)))
    i0 = n - k + 1
    df   = df[i0:n, :]
    step = i0:n                      # original step numbers on the x-axis, not 1-based

    comps = [c[6:end] for c in names(df) if startswith(c, "rmse_")]
    isempty(comps) && error("No rmse_* metric columns found in $(joinpath(path, file)).")

    panels = []
    for comp in comps
        push!(panels, Plots.plot(step, df[!, "rmse_"*comp];
            ylabel = "RMSE ($comp)", yscale = :log10, lw = 2, legend = false))
        pb = Plots.plot(step, df[!, "bias_"*comp];
            ylabel = "bias ($comp)", lw = 2, legend = false)
        Plots.hline!(pb, [0]; color = :black, ls = :dash, lw = 1, label = "")
        push!(panels, pb)
    end

    Plots.xlabel!(panels[end-1], "Training step")
    Plots.xlabel!(panels[end],   "Training step")

    return Plots.plot(panels...; layout = (length(comps), 2),
                      plot_title = "Per-component metrics", kwargs...)
end