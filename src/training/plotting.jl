### Plotting of training runs
###
### Functions for plotting training data
###     - plot_loss:            simple loss plot over training steps
###     - plot_training:        plots loss and also parameter- and gradient norm over training steps
###     - plot_training_comp:   plots loss comparison of two training runs
###     - plot_metrics_norm:    plots normalized metrics (loss, rmse, bias) from a training .csv file
###     - plot_metrics_raw:     plots raw metrics (rmse, bias) from a training .csv file



# One colour per field, shared by every training and rollout plot
const FIELD_COLORS = (;
    total = :black,
    T     = :firebrick,
    olw   = :steelblue,
    slwd  = :rebeccapurple,
    slwu  = :darkorange,
)

# Utility function for not-breaking the axis for log10 plots
log_or_lin(v) = all(>(0), v) ? :log10 : :identity

# Utility function for plotting vertical lines at the end of each initial condition
ic_bounds(ic::AbstractVector) = [i + 0.5 for i in 1:length(ic)-1 if ic[i] != ic[i+1]]



# Plot timeseries of loss of a training run
function plot_loss(loss::AbstractVector; kwargs...)

    return Plots.plot(
        loss;
        xlabel = "Training step",
        ylabel = "Loss",
        yscale = log_or_lin(loss),
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
    ic = nothing,
)

    bounds = isnothing(ic) ? Float64[] : ic_bounds(ic)

    # Mean over consecutive blocks of n_batch steps, and their x-positions
    bmean(v) = [Statistics.mean(@view v[i:min(i+n_batch-1, lastindex(v))]) for i in 1:n_batch:length(v)]
    xb = [min(i+n_batch-1, length(loss)) for i in 1:n_batch:length(loss)]


    # Loss plot
    p1 = Plots.plot(
        loss;
        ylabel = "Loss",
        yscale = log_or_lin(loss),
        legend = false, 
        loss_kwargs...,
    )
    Plots.plot!(p1, xb, bmean(loss); label = "batch mean", lw = 2, color = :black)
    isempty(bounds) || Plots.vline!(p1, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    # Parameter norm plot
    p2 = Plots.plot(
        pnorm;
        ylabel = "Parameter norm",
        legend = false,  
        pnorm_kwargs...,
    )
    Plots.plot!(p2, xb, bmean(pnorm); label = "batch mean", lw = 2, color = :black)
    isempty(bounds) || Plots.vline!(p2, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    # Gradient norm plot
    p3 = Plots.plot(
        gnorm;
        xlabel="Training step",
        ylabel="Gradient norm",
        yscale=log_or_lin(gnorm),
        legend = false,  
        gnorm_kwargs...,
    )
    Plots.plot!(p3, xb, bmean(gnorm); label = "batch mean", lw = 2, color = :black)
    isempty(bounds) || Plots.vline!(p3, bounds; color = :gray, ls = :dot, lw = 1, label = "")

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
        yscale = log_or_lin(reduce(vcat, losses)),
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
    bounds = ic_bounds(df.ic)

    # Weights are not stored in the .csv, so they have to be passed in
    w    = isnothing(weights) ? (; T = 1, olw = 1, slwd = 1) : weights


    # Losses: total and per-field contributions
    p1 = Plots.plot(w.T .* df.loss_T ./ df.loss_total;
        ylabel = "loss share", ylims = (0, 1), lw = 2, label = "T", color = FIELD_COLORS.T)
    Plots.plot!(p1, w.olw  .* df.loss_olw  ./ df.loss_total; lw = 2, label = "olw",  color = FIELD_COLORS.olw)
    Plots.plot!(p1, w.slwd .* df.loss_slwd ./ df.loss_total; lw = 2, label = "slwd", color = FIELD_COLORS.slwd)
    Plots.vline!(p1, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    # Normalized rmse (in units of the field's std)
    p2 = Plots.plot(df.nrmse_T;
        ylabel = "norm. RMSE [T_scale/_σ]", yscale = log_or_lin(df.nrmse_T), lw = 2,
        label = "T", color = FIELD_COLORS.T)
    Plots.plot!(p2, df.nrmse_olw;  lw = 2, label = "olw",  color = FIELD_COLORS.olw)
    Plots.plot!(p2, df.nrmse_slwd; lw = 2, label = "slwd", color = FIELD_COLORS.slwd)
    Plots.vline!(p2, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    # Normalized bias (in units of the field's std)
    p3 = Plots.plot(df.nbias_T;
        xlabel = "Training step", ylabel = "norm. bias [T_scale/_σ]", lw = 2,
        label = "T", color = FIELD_COLORS.T)
    Plots.plot!(p3, df.nbias_olw;  lw = 2, label = "olw",  color = FIELD_COLORS.olw)
    Plots.plot!(p3, df.nbias_slwd; lw = 2, label = "slwd", color = FIELD_COLORS.slwd)
    Plots.hline!(p3, [0]; color = :black, ls = :dash, lw = 1, label = "")
    Plots.vline!(p3, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    defaults = (; size = (700, 900), left_margin = 8Plots.mm)

    return Plots.plot(p1, p2, p3; layout = (3, 1), merge(defaults, plot_kwargs)...)
end



# Plot raw metrics of a training run in physical units, grouped by unit
function plot_metrics_raw(;  dir="", file="", plot_kwargs = (;))

    df = csv_read(; dir, file)
    bounds = ic_bounds(df.ic)

    # Temperature [K]
    p1 = Plots.plot(df.rmse_T; ylabel = "T [K]", lw = 2, label = "RMSE", color = FIELD_COLORS.T)
    Plots.plot!(p1, df.bias_T; lw = 2, label = "bias", color = FIELD_COLORS.T, ls = :dash)
    Plots.hline!(p1, [0]; color = :black, ls = :dash, lw = 1, label = "")
    Plots.vline!(p1, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    # Fluxes [W/m²] — same unit, so they share a panel
    p2 = Plots.plot(df.rmse_olw;  xlabel = "Training step", ylabel = "Flux [W/m²]",
                    lw = 2, label = "RMSE olw",  color = FIELD_COLORS.olw)
    Plots.plot!(p2, df.bias_olw;  lw = 2, label = "bias olw",  color = FIELD_COLORS.olw,  ls = :dash)
    Plots.plot!(p2, df.rmse_slwd; lw = 2, label = "RMSE slwd", color = FIELD_COLORS.slwd)
    Plots.plot!(p2, df.bias_slwd; lw = 2, label = "bias slwd", color = FIELD_COLORS.slwd, ls = :dash)
    Plots.hline!(p2, [0]; color = :black, ls = :dash, lw = 1, label = "")
    Plots.plot!(p2, df.bias_C; lw = 2, label = "bias C (= -olw -slwd)", color = :black, ls = :dot)
    Plots.vline!(p2, bounds; color = :gray, ls = :dot, lw = 1, label = "")

    defaults = (; size = (700, 600), left_margin = 8Plots.mm)
    return Plots.plot(p1, p2; layout = (2, 1), merge(defaults, plot_kwargs)...)
end