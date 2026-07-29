### Functions for skill and rollout evaluation
###
### XXX





# XXX REMOVE Plot skill curves for every scheme in respect to the metric
function plot_rollout(;
    results::NamedTuple,
    metric::Symbol,
    var::Symbol = :temperature,
    title::String = "",
)

    # Create empty canvas
    p = Plots.plot(; 
        xlabel = "forecast horizon [days]", 
        ylabel = "$(var) $(metric)",
        legend = :topleft, 
        title = title
    )

    # Plot the lines for the schemes
    for (name, r) in pairs(results)
        Plots.plot!(p, collect(r.days), r.curve[var][metric];
                    label = String(name), lw = 2)
    end
    
    return p
end