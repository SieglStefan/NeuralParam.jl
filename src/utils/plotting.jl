### Plotting utilities
###
### Helper functions for visualizing losses, training diagnostics,
### temperature-field comparisons, and heatmaps.



# Plot training diagnostics for NeuralLinearLongwave variants
function plot_training(
    L, PN, GN;
    loss_kwargs = (;),
    pnorm_kwargs = (;),
    gnorm_kwargs = (;),
    plot_kwargs = (;),
)
    p1 = Plots.plot(
        0:length(L)-1,
        L;
        yscale = :log10,
        ylabel = "Loss",
        title = "Loss evolution",
        loss_kwargs...,
    )

    p2 = Plots.plot(
        0:length(PN)-1,
        PN;
        yscale = :log10,
        ylabel = "|p|",
        title = "Parameter norm",
        pnorm_kwargs...,
    )

    p3 = Plots.plot(
        0:length(GN)-1,
        GN;
        yscale = :log10,
        xlabel = "Training step",
        ylabel = "|g|",
        title = "Gradient norm",
        gnorm_kwargs...,
    )

    p = Plots.plot(p1, p2, p3; layout = (3, 1), plot_kwargs...)
    display(p)

    return p
end



# Plot only the loss
function plot_loss(L; kwargs...)
    p = Plots.plot(
        0:length(L)-1,
        L;
        yscale = :log10,
        xlabel = "Step",
        ylabel = "Loss",
        title = "Loss evolution",
        kwargs...,
    )

    display(p)

    return p
end



# Plot metric-based differences between target and comparison trajectories
function plot_comparison(
    T_target,
    T_comp;
    metric = rmse,
    Δt_sec,
    labels = nothing,
    kwargs...,
)
    # Time axis in days
    t_days = (0:length(T_target)-1) .* Δt_sec ./ (60 * 60 * 24)

    # Default labels
    labels = labels === nothing ? ["T_comp_$i" for i in eachindex(T_comp)] : labels

    p = Plots.plot(;
        xlabel = "Time (days)",
        ylabel = uppercase(string(metric)),
        title = "Temperature evolution comparison",
        legend = :topleft,
        kwargs...,
    )

    for (i, comp) in enumerate(T_comp)
        values = metric.(T_target, comp)
        Plots.plot!(p, t_days, values; label = labels[i])
    end

    display(p)

    return p
end



# Plot a single heatmap
function plot_heatmap(
    F;
    title = "Heatmap",
    kwargs...,
)
    fig = CairoMakie.heatmap(
        F;
        title,
        kwargs...,
    )

    display(fig)

    return fig
end



# Plot several heatmaps with common colorbar scaling
function plot_heatmaps(
    F_vec;
    titles = nothing,
    kwargs...,
)
    cmin = minimum(minimum.(F_vec))
    cmax = maximum(maximum.(F_vec))
    crange = (cmin, cmax)

    titles = titles === nothing ? ["Heatmap $i" for i in eachindex(F_vec)] : titles

    fig_vec = []

    for (i, F) in enumerate(F_vec)
        fig = plot_heatmap(
            F;
            title = titles[i],
            colorrange = crange,
            kwargs...,
        )

        push!(fig_vec, fig)
    end

    return fig_vec
end