### Logging of training data
###
### Functions for logging training data to .csv files and printing run configuration information



# Initialize .csv file and create a info.toml file for meta data
function csv_init(metric_keys; dir="", file="")

    # Create path and put together filepath
    mkpath(dir)
    filepath = joinpath(dir, file)

    # Write header
    open(filepath, "w") do io
        println(io, "ic,traj,n_steps,eta," * join(metric_keys, ","))
    end

    # Print information
    @info ".csv file created and initialized at $(filepath)!"

    return filepath
end


# Write a row of training data to .csv
function csv_row!(
    ic, traj, n_steps, eta;
    metrics,
    dir="", file=""
)

    # Put together filepath
    filepath = joinpath(dir, file)

    # Write a data row
    open(filepath, "a") do io
        println(io, join((ic, traj, n_steps, eta, values(metrics)...), ","))
    end

    return nothing
end


# Read .csv data for plotting
function csv_read(; dir="", file="")

    # Put together filepath
    filepath = joinpath(dir, file)

    # Read .csv data
    return CSV.read(filepath, DataFrame; comment="#")
end



# Function for calculating metrics (losses, norms, bias,...) for logging
function compute_metrics(lw_train, tc, sims, grads)
    
    # Unpack loss weights, field normalizations and grid weights
    w = tc.loss_config.weights
    fn = tc.loss_config.field_norm
    gw = tc.loss_config.grid_weights


    # Target fields
    T_target = SpeedyWeather.get_step(sims.target.variables.grid.temperature)
    olw_target = sims.target.variables.parameterizations.outgoing_longwave
    slwd_target = sims.target.variables.parameterizations.surface_longwave_down

    # Training fields
    T_train = SpeedyWeather.get_step(sims.train.variables.grid.temperature)
    olw_train = sims.train.variables.parameterizations.outgoing_longwave
    slwd_train = sims.train.variables.parameterizations.surface_longwave_down

    # Residuals
    res_T    = T_train    .- T_target
    res_olw  = olw_train  .- olw_target
    res_slwd = slwd_train .- slwd_target


    # Normalized rmse (dimensionless)
    nrmse_T    = sqrt(wmean(abs2.(res_T    ./ fn.T),    gw))
    nrmse_olw  = sqrt(wmean(abs2.(res_olw  ./ fn.olw),  gw))
    nrmse_slwd = sqrt(wmean(abs2.(res_slwd ./ fn.slwd), gw))

    # Calculate total loss
    loss_t, loss_olw, loss_slwd = nrmse_T^2, nrmse_olw^2, nrmse_slwd^2
    loss_total = w.T * loss_t + w.olw * loss_olw + w.slwd * loss_slwd


    # Return diagnostics
    return (;
    
        # Losses
        loss_total = loss_total, loss_T = loss_t, loss_olw = loss_olw, loss_slwd = loss_slwd,

        # Raw rmse and bias (physical units: K, W/m^2)
        rmse_T    = sqrt(wmean(abs2.(res_T),    gw)),   bias_T    = wmean(res_T,    gw),
        rmse_olw  = sqrt(wmean(abs2.(res_olw),  gw)),   bias_olw  = wmean(res_olw,  gw),
        rmse_slwd = sqrt(wmean(abs2.(res_slwd), gw)),   bias_slwd = wmean(res_slwd, gw),

        # Normalized rmse and bias
        nrmse_T,    nbias_T    = wmean(res_T    ./ fn.T,    gw),
        nrmse_olw,  nbias_olw  = wmean(res_olw  ./ fn.olw,  gw),
        nrmse_slwd, nbias_slwd = wmean(res_slwd ./ fn.slwd, gw),

        # Typical target values (area-weighted mean) — scale to judge rmse/bias against
        mean_T    = wmean(T_target,    gw),
        mean_olw  = wmean(olw_target,  gw),
        mean_slwd = wmean(slwd_target, gw),
        
        # Optimizer diagnostics
        pnorm = tree_l2norm(lw_train.ps),
        gnorm = tree_l2norm(grads),
    )
end



# Print information about the run configuration
function print_config(tc, dt_sec)

    # single time step in days
    dt_day = dt_sec /3600 /24

    # Total updates per training
    up_total = tc.n_ic * tc.n_traj ÷ tc.n_batch

    # Batch window and gap size in days
    t_batch = tc.n_batch * dt_day
    t_gap = tc.n_gap * dt_day

    # Min. and max. length of single gradient trajectory in days
    t_grad_start = tc.n_steps_0 * dt_day
    t_grad_end = (tc.n_steps_0 + tc.n_steps_inc * (tc.n_ic-1)) * dt_day

    # Total training period per IC in days
    t_train_start = tc.n_traj * (t_grad_start + t_gap)
    t_train_end = tc.n_traj * (t_grad_end + t_gap)

    # Print information
    println("----------Training run configuration:----------")
    println("  - Number of total updates: ", up_total)
    println("  - Length of training period per IC (days): \t\tStart: ", t_train_start, "\tEnd: ", t_train_end)
    println("  - Length of batch window (days): ", t_batch)
    println("  - Length of gap (days): ", t_gap)
    println("  - Length of trajectory differentiation (hours): \tStart: ", t_grad_start*24, "\tEnd: ", t_grad_end*24)
    println("-----------------------------------------------")
end