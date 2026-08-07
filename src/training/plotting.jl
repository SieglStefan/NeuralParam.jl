### Plotting of training runs
###
### Functions for plotting training data
###     - plot_loss:            simple loss plot over training steps
###     - plot_training:        plots loss and also parameter- and gradient norm over training steps
###     - plot_training_comp:   plots loss comparison of two training runs
###     - plot_metrics:         plots per-component metrics from a training .csv file



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
function plot_loss(; dir="", file="", kwargs...)
    
    # Read and extract data
    df = csv_read(;dir, file)
    loss = df.loss_total
    
    return plot_loss(loss; kwargs...)
end



# Plot timeseries of loss, parameter- and gradient norm of a training run
function plot_training(
    loss::AbstractVector,
    pnorm::AbstractVector,
    gnorm::AbstractVector;
    n_batch = 1,
    loss_kwargs = (;),
    pnorm_kwargs = (;),
    gnorm_kwargs = (;),
    plot_kwargs = (;),
)

    # Mean over consecutive blocks of n_batch steps, and their x-positions
    bmean(v) = [Statistics.mean(@view v[i:min(i+n_batch-1, lastindex(v))]) for i in 1:n_batch:length(v)]
    xb = [min(i+n_batch-1, length(loss)) for i in 1:n_batch:length(loss)]


    # Loss plot
    p1 = Plots.plot(
        loss;
        ylabel = "Loss",
        yscale = :log10,
        legend = false, 
        loss_kwargs...,
    )
    Plots.plot!(p1, xb, bmean(loss); label = "batch mean", lw = 2, color = :black)

    # Parameter norm plot
    p2 = Plots.plot(
        pnorm;
        ylabel = "Parameter norm",
        legend = false,  
        pnorm_kwargs...,
    )
    Plots.plot!(p2, xb, bmean(pnorm); label = "batch mean", lw = 2, color = :black)

    # Gradient norm plot
    p3 = Plots.plot(
        gnorm;
        xlabel="Training step",
        ylabel="Gradient norm",
        yscale=:log10,
        legend = false,  
        gnorm_kwargs...,
    )
    Plots.plot!(p3, xb, bmean(gnorm); label = "batch mean", lw = 2, color = :black)

    defaults = (; size = (600, 900), left_margin = 8Plots.mm)
    merged = merge(defaults, plot_kwargs)

    return Plots.plot(p1, p2, p3; layout=(3, 1), merged...)
end

# Plot timeseries of loss, parameter- and gradient norm of a training run from file
function plot_training(; dir="", file="", n_batch, kwargs...)
    
    # Read and extract data
    df = csv_read(;dir, file)

    loss = df.loss_total
    pnorm = df.pnorm
    gnorm = df.gnorm
    
    return plot_training(loss, pnorm, gnorm; n_batch, kwargs...)
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
    losses = [csv_read(; dir=p, file=f).loss_total for (p,f) in runs]
    labs = isnothing(labels) ? [f for (_,f) in runs] : labels

    return plot_training_comp(losses; labels=labs, kwargs...)
end  



# Plot normalized metrics of a training run: losses, normalized rmse and bias
function plot_metrics_norm(; dir="", file="", weights=nothing, plot_kwargs = (;))

    df = csv_read(; dir, file)

    # Weights are not stored in the .csv, so they have to be passed in
    w    = isnothing(weights) ? (; T = 1, olw = 1, slwd = 1) : weights
    wlab = isnothing(weights) ? "unweighted" : "weighted"


    # Losses: total and per-field contributions
    p1 = Plots.plot(df.loss_total;
        ylabel = "Loss", yscale = :log10, lw = 2, label = "total", color = :black)
    Plots.plot!(p1, w.T    .* df.loss_T;    lw = 2, label = "T ($wlab)")
    Plots.plot!(p1, w.olw  .* df.loss_olw;  lw = 2, label = "olw ($wlab)")
    Plots.plot!(p1, w.slwd .* df.loss_slwd; lw = 2, label = "slwd ($wlab)")

    # Normalized rmse (in units of the field's std)
    p2 = Plots.plot(df.nrmse_T;
        ylabel = "norm. RMSE [σ]", yscale = :log10, lw = 2, label = "T")
    Plots.plot!(p2, df.nrmse_olw;  lw = 2, label = "olw")
    Plots.plot!(p2, df.nrmse_slwd; lw = 2, label = "slwd")

    # Normalized bias (in units of the field's std)
    p3 = Plots.plot(df.nbias_T;
        xlabel = "Training step", ylabel = "norm. bias [σ]", lw = 2, label = "T")
    Plots.plot!(p3, df.nbias_olw;  lw = 2, label = "olw")
    Plots.plot!(p3, df.nbias_slwd; lw = 2, label = "slwd")
    Plots.hline!(p3, [0]; color = :black, ls = :dash, lw = 1, label = "")


    defaults = (; size = (700, 900), left_margin = 8Plots.mm)

    return Plots.plot(p1, p2, p3; layout = (3, 1), merge(defaults, plot_kwargs)...)
end



# Plot raw metrics of a training run in physical units, grouped by unit
function plot_metrics_raw(;  dir="", file="", plot_kwargs = (;))

    df = csv_read(; dir, file)

    # Temperature [K]
    p1 = Plots.plot(df.rmse_T; ylabel = "T [K]", lw = 2, label = "RMSE")
    Plots.plot!(p1, df.bias_T; lw = 2, label = "bias")
    Plots.hline!(p1, [0]; color = :black, ls = :dash, lw = 1, label = "")

    # Fluxes [W/m²] — same unit, so they share a panel
    p2 = Plots.plot(df.rmse_olw;
        xlabel = "Training step", ylabel = "Flux [W/m²]", lw = 2, label = "RMSE olw")
    Plots.plot!(p2, df.bias_olw;  lw = 2, label = "bias olw")
    Plots.plot!(p2, df.rmse_slwd; lw = 2, label = "RMSE slwd")
    Plots.plot!(p2, df.bias_slwd; lw = 2, label = "bias slwd")
    Plots.hline!(p2, [0]; color = :black, ls = :dash, lw = 1, label = "")


    defaults = (; size = (700, 600), left_margin = 8Plots.mm)
    return Plots.plot(p1, p2; layout = (2, 1), merge(defaults, plot_kwargs)...)
end