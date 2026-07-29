### Functions for skill and rollout evaluation
###
### XXX



# Reduce one curve over trajectories: MAX_HORIZON × N_TRAJ -> vector
rollout_curve(r, var, metric; f = mean) = vec(f(r.curve[var][metric]; dims = 2))


function plot_rollout(; rollouts::NamedTuple, vars, metrics, ribbon = true, kwargs...)

    panels = []

    # One panel per variable and metric
    for var in vars, metric in metrics
        p = Plots.plot(; xlabel = "forecast horizon [days]",
                         ylabel = "$(var) $(metric)", legend = :topleft)

        # One line per scheme
        for (name, r) in pairs(rollouts)
            rib = ribbon ? rollout_curve(r, var, metric; f = std) : nothing
            Plots.plot!(p, collect(r.days), rollout_curve(r, var, metric);
                        label = String(name), lw = 2, ribbon = rib, fillalpha = 0.15)
        end

        push!(panels, p)
    end

    return Plots.plot(panels...;
        layout = (length(vars), length(metrics)),
        size   = (500 * length(metrics), 350 * length(vars)),
        kwargs...)
end


# Heatmap field of one rollout: j indexes heatmap_days, layer optional (2D vars have none)
hm_field(r, var, j; layer = nothing) =
    isnothing(layer) ? r.heatmap_states[var][j] : r.heatmap_states[var][j][:, layer]

# Symmetric color range around zero, for difference maps
sym_range(fields) = (m = maximum(maximum(abs, vec(f)) for f in fields); (-m, m))


### Heatmaps of all rollouts, one figure per heatmap day
### ref = nothing -> absolute fields (is the structure preserved?)
### ref = :Name   -> difference to that rollout (where do the schemes differ?)
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