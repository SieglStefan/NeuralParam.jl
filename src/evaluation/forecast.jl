### Functions for skill and rollout evaluation
###
### XXX



function evaluate_forecast(;
    scheme,                                     # evaluated scheme
    reference;                                  # comparison reference
    spectral_grid,          
    model_type,             # used model for schemes
    max_horizon,                           # maximum forecast length in days
    n_traj,                                 # number of trajectories sampled
    metrics::NamedTuple = (; rmse, bias),       # used metrics
    heatmap_days,                          # list of days data for heatmaps is returned
    heatmap_traj,                           # choice of specific trajectory
)

    # Extract time stepping
    Δt_sec = initialize!(model_type(spectral_grid; longwave_radiation = scheme)).model.time_stepping.Δt_sec

    # Calculate number of steps for one day
    n_steps_day = steps_from_days(1, Δt_sec)

    # Calculate starting days
    start_days = round.(Int, range(0, 365, length = n_traj + 1))[1:end-1]


    # Prepare container for metrics
    sums = map(_ -> zeros(Float64, max_horizon), metrics)       # = (; rmse=[0,0,...],  bias=[0,0,...], ...)

    # Prepare container for heatmap data
    kept_traj = nothing


    # Loop over all starting days in a year e.g.: (0,7,14,...)
    for (i,s) in enumerate(start_days)

        # Initialize simulation
        sim  = initialize!(model_type(spectral_grid; longwave_radiation = scheme))

        # Sample trajectory
        traj = sample_trajectory(
            sim, 
            reference[s + 1]; 
            n_steps = max_horizon * n_steps_day,
            n_gap = n_steps_day                     # samplee once every day
        )


        # Loop over horizons
        for h in 1:max_horizon
            f = traj.temperature[h + 1]                     # sampled trajectroy temperature
            t = reference[s + h + 1].grid.temperature       # reference temperature at day s+h
            
            # Calculate metric and add to sums
            for (name, metric) in pairs(metrics)
                sums[name][h] += metric(f, t)
            end
        end

        # Filter heatmap days
        if i == heatmap_traj
            kept_traj = [traj.temperature[d+1] for d in heatmap_days]
        end
    end

    # Calculate mean in respect to all starting days
    forecast_stats = map(v -> v ./ length(start_days), sums)


    return (; days = 1:max_horizon, curve = forecast_stats, heatmaps = kept_traj)
end


# Wrapper for a list (NamedTuple) of schemes
function evaluate_forecast(schemes::NamedTuple, reference; kwargs...)
    return map(scheme -> evaluate_forecast(scheme, reference; kwargs...), schemes)
end



# Plot skill curves for every scheme in respect to the metric
function plot_forecast(;
    results::NamedTuple,
    metric::Symbol,
    title::String = "",
)

    # Create empty canvas
    p = Plots.plot(; 
        xlabel = "forecast horizon [days]", 
        ylabel = String(metric),
        legend = :topleft, 
        title = title
    )

    # Plot the lines for the schemes
    for (name, r) in pairs(results)
        Plots.plot!(p, collect(r.days), r.curve[metric];
                    label = String(name), lw = 2)
    end
    return p
end