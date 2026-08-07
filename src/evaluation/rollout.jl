### Functions for skill and rollout evaluation
###
### Evaluates rollouts of trained schemes and compares them to reference data sets, including skill scores and heatmaps of fields



# Reduce the column dimension of one metric
#   - rmse combines in quadrature, maxdiff takes the worst layer, everything else averages
reduce_cols(a, metric) =
    metric === :rmse    ? sqrt.(mean(abs2, a; dims = 3)) :      # combines in quadrature
    metric === :maxdiff ? maximum(a; dims = 3)           :      # takes the worst layer
                          mean(a; dims = 3)                     # rest averages



# Reduce one curve over trajectories: (day, traj, col) -> vector over lead days
#   - layer = nothing reduces all column entries, an integer picks one vertical layer
function rollout_curve(r, probe, metric; layer = nothing, f = mean)

    # Pick the metric and reduce the column dimension
    a = r.scores[probe][metric]
    a = isnothing(layer) ? reduce_cols(a, metric) : a[:, :, layer:layer]

    # Reduce over trajectories
    return vec(f(dropdims(a; dims = 3); dims = 2))
end


# Plot ONE metric for several probed fields, one row per field (3x1 for T, olw, slwd)
#   - one line per scheme, ribbon = spread over the trajectories
#   - slwu is probed but not plotted by default: it is not part of the loss
function plot_rollout(;
    rollouts::NamedTuple,
    metric::Symbol,
    probes = (:T, :olw, :slwd),
    layer = nothing,
    ribbon = true,
    ncols = 1,
    kwargs...
)

    # One panel per probed field
    panels = map(collect(probes)) do probe

        unit = get(DEF_PROBE_UNITS, probe, "")
        p = Plots.plot(; xlabel = "forecast horizon [days]",
                         ylabel = isempty(unit) ? String(probe) : "$(probe) [$(unit)]",
                         title  = String(probe), titlefontsize = 10, legend = :topleft)

        # One line per scheme
        for (name, r) in pairs(rollouts)
            rib = ribbon ? rollout_curve(r, probe, metric; layer, f = std) : nothing
            Plots.plot!(p, collect(r.days), rollout_curve(r, probe, metric; layer);
                        label = String(name), lw = 2, ribbon = rib, fillalpha = 0.15)
        end

        return p
    end

    # Arrange in a grid, one column by default -> 3x1 for the three loss fields
    nrows = cld(length(panels), ncols)
    lay   = isnothing(layer) ? "all layers" : "layer $(layer)"

    return Plots.plot(panels...;
        layout     = (nrows, ncols),
        size       = (650 * ncols, 330 * nrows),
        plot_title = "$(metric) — $(lay)",
        kwargs...)
end


# One figure per metric, keyed by metric name; stored as .png if dir is given
function plot_rollout_metrics(;
    rollouts::NamedTuple,
    metrics = (:rmse, :bias, :maxdiff, :mean),
    dir = nothing,
    kwargs...
)

    # One figure per metric
    figs = (; (m => plot_rollout(; rollouts, metric = m, kwargs...) for m in metrics)...)

    # Save one .png per metric
    if !isnothing(dir)
        mkpath(dir)
        for (name, fig) in pairs(figs)
            Plots.savefig(fig, joinpath(dir, "rollout_$(name).png"))
        end
        @info "Rollout plots stored at $(dir)!"
    end

    return figs
end



### Heatmaps of all rollouts, one figure per heatmap day
# Heatmap field of one rollout: j indexes heatmap_days, layer optional (2D vars have none)
hm_field(r, var, j; layer = nothing) =
    isnothing(layer) ? r.heatmap_states[var][j] : r.heatmap_states[var][j][:, layer]




function plot_rollout_heatmaps(;
    rollouts::NamedTuple,
    var::Symbol = :T,               # a PROBE name, heatmap_states is keyed by probes
    layer = nothing,
    ref = nothing,
    colorrange = nothing,           # nothing = derive one common range over all days
    kwargs...
)

    # All rollouts of one protocol share the heatmap days
    days = first(values(rollouts)).heatmap_days

    # Difference to itself is zero everywhere, so drop the reference panel
    names = isnothing(ref) ? collect(keys(rollouts)) : collect(filter(!=(ref), keys(rollouts)))

    lay = isnothing(layer) ? "" : ", layer $(layer)"


    # Collect the fields of every day FIRST, so all figures can share one colorbar
    #   - with ref given, plot the difference to that rollout instead
    fields_per_day = map(eachindex(days)) do j

        fields = [hm_field(rollouts[n], var, j; layer) for n in names]

        isnothing(ref) && return fields

        ref_f = hm_field(rollouts[ref], var, j; layer)
        return [f .- ref_f for f in fields]
    end

    # One common colorrange over every day, so the figures are comparable
    all_fields = reduce(vcat, fields_per_day)
    crange = !isnothing(colorrange) ? colorrange :
             isnothing(ref)         ? finite_range(all_fields) : sym_range(all_fields)

    # Diverging colormap for difference maps
    cmap = isnothing(ref) ? (;) : (; colormap = :balance)


    # One figure per heatmap day
    return map(enumerate(days)) do (j, d)

        title = isnothing(ref) ? "$(var)$(lay) — day $(d)" :
                                 "$(var)$(lay) — day $(d), difference to $(ref)"

        style = (; colorrange = crange, cmap..., suptitle = title)

        plot_heatmaps(fields_per_day[j]; titles = String.(names), merge(style, values(kwargs))...)
    end
end