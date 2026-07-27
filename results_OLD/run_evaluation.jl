### Functions for running evaluation
###
### XXX



# Run a skill evaluation
function run_evaluation_forecast(
    schemes,
    reference;
    spectral_grid,
    model_type,
    max_horizon,
    n_traj,
    heatmap_days,
    heatmap_traj,           
    heatmap_layer,           
    run_dir,
    folder_name,
)  

    # Create folder for skill evaluation
    out_dir = fresh_out_dir(run_dir, folder_name)


    ### Compute rmse and bias for different lead times
    results = evaluate_forecast(
        schemes, 
        reference;
        spectral_grid, 
        model_type,
        max_horizon, 
        n_traj,
        heatmap_days,
    )


    ### Create and store plots
    # rmse
    p_rmse = plot_forecast(;results, metric = :rmse)
    Plots.title!(p_rmse, "RMSE Trend")

    # bias
    p_bias = plot_forecast(;results, metric = :bias)
    Plots.title!(p_bias, "Bias Trend")

    # Save plots
    Plots.savefig(p_rmse, joinpath(out_dir, "rmse.png"))
    Plots.savefig(p_bias,  joinpath(out_dir, "bias.png"))


    ### Create and store heatmaps
    # Check if there are any days to plot
    if !isempty(heatmap_days)

        # Create folder for heatmaps
        hm_dir = joinpath(out_dir, "heatmaps")
        mkpath(hm_dir)

        # Create heatmap titles out of scheme keys
        titles = [String(k) for k in keys(results)]

        for layer in heatmap_layer

            # Shared colorbar calculated from first results scheme (target mostly)
            crange = extrema(reduce(vcat, (vec(f[:, layer]) for f in first(results).heatmaps)))

            # Create and save heatmaps
            for (j, d) in enumerate(heatmap_days)
                fields = [r.heatmaps[j][:, layer] for r in results]
                fig = plot_heatmaps(fields; titles = titles, colorrange = crange)
                CairoMakie.save(joinpath(hm_dir, "day$(d)_layer$(layer).png"), fig)
            end
        end
    end

    return nothing
end





function run_evaluation_benchmark(
    schemes;
    spectral_grid,
    model_type,
    n_steps,
    run_dir,
)

    # Create folder for benchmark evaluation
    out_dir = fresh_out_dir(run_dir, "benchmark")

    # Run benchmark
    results = evaluate_benchmark(
        schemes;
        spectral_grid,
        model_type,
        n_steps
    )
    

    # Saves the benchmark results in a small table
    write_info(; 
        path = out_dir, 
        file = "benchmark.toml",
        (name => Dict(  "ms_per_step"     => r.per_step_ms,
                        "kb_per_step"     => r.per_step_kb,
                        "allocs_per_step" => r.per_step_allocs)
            for (name, r) in pairs(results)
        )...
    )
    
    return nothing
end



# XXX
function run_evaluation_ab(schemes; nlayers, run_dir, folder_name = "ab")
    out_dir = fresh_out_dir(run_dir, folder_name)

    fig_ab = plot_ab_profiles(schemes; nlayers, title = "a and b profiles per scheme")
    Plots.savefig(fig_ab, joinpath(out_dir, "ab.png"))

    fig_dT = plot_dT_profile(schemes; nlayers, title = "Effective heating dT for a typical profile")
    Plots.savefig(fig_dT, joinpath(out_dir, "dT.png"))

    return nothing
end



