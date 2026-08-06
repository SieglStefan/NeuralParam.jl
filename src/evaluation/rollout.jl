### Functions for skill and rollout evaluation
###
### Evaluates rollouts of trained schemes and compares them to reference data sets, including skill scores and heatmaps of fields



# Reduce the column dimension of one metric: rmse combines in quadrature, everything else averages
reduce_cols(a, metric) = metric === :rmse ? sqrt.(mean(abs2, a; dims = 3)) : mean(a; dims = 3)

# Reduce one curve over trajectories: (day, traj, col) -> vector over lead days
#   - layer = nothing reduces all column entries, an integer picks one vertical layer
function rollout_curve(r, probe, metric; layer = nothing, f = mean)

    # Pick the metric and reduce the column dimension
    a = r.err[probe][metric]
    a = isnothing(layer) ? reduce_cols(a, metric) : a[:, :, layer:layer]

    # Reduce over trajectories
    return vec(f(dropdims(a; dims = 3); dims = 2))
end


# Plot rollout curves of multiple schemes, one panel per probed field and metric
function plot_rollout(; rollouts::NamedTuple, probes, metrics, layer = nothing, ribbon = true, kwargs...)

    panels = []

    # One panel per probed field and metric
    for probe in probes, metric in metrics
        p = Plots.plot(; xlabel = "forecast horizon [days]",
                         ylabel = "$(probe) $(metric)", legend = :topleft)

        # One line per scheme
        for (name, r) in pairs(rollouts)
            rib = ribbon ? rollout_curve(r, probe, metric; layer, f = std) : nothing
            Plots.plot!(p, collect(r.days), rollout_curve(r, probe, metric; layer);
                        label = String(name), lw = 2, ribbon = rib, fillalpha = 0.15)
        end

        push!(panels, p)
    end

    return Plots.plot(panels...;
        layout = (length(probes), length(metrics)),
        size   = (500 * length(metrics), 350 * length(probes)),
        kwargs...)
end


# Heatmap field of one rollout: j indexes heatmap_days, layer optional (2D vars have none)
hm_field(r, var, j; layer = nothing) =
    isnothing(layer) ? r.heatmap_states[var][j] : r.heatmap_states[var][j][:, layer]

# Symmetric color range around zero, for difference maps
sym_range(fields) = (m = maximum(maximum(abs, vec(f)) for f in fields); (-m, m))


### Heatmaps of all rollouts, one figure per heatmap day
function plot_rollout_heatmaps(;
    rollouts::NamedTuple,
    var::Symbol = :temperature,
    layer = nothing,
    ref = nothing,
    kwargs...
)

    # All rollouts of one protocol share the heatmap days
    days = first(values(rollouts)).heatmap_days

    # Difference to itself is zero everywhere, so drop the reference panel
    names = isnothing(ref) ? collect(keys(rollouts)) : collect(filter(!=(ref), keys(rollouts)))

    lay = isnothing(layer) ? "" : ", layer $(layer)"

    return map(enumerate(days)) do (j, d)

        fields = [hm_field(rollouts[n], var, j; layer) for n in names]

        # Subtract the reference field and switch to a diverging colormap
        if isnothing(ref)
            style = (; suptitle = "$(var)$(lay) — day $(d)")
        else
            ref_f  = hm_field(rollouts[ref], var, j; layer)
            fields = [f .- ref_f for f in fields]
            style  = (; colorrange = sym_range(fields), colormap = :balance,
                        suptitle = "$(var)$(lay) — day $(d), difference to $(ref)")
        end

        plot_heatmaps(fields; titles = String.(names), merge(style, values(kwargs))...)
    end
end