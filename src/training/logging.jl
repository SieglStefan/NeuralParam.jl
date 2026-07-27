# Intialize .csv file for logging training data and create a info.toml file for meta data
function csv_init(metric_keys; path="", file="")

    # Create path and put together filepath
    mkpath(path)
    filepath = joinpath(path, file)

    # Write header
    open(filepath, "w") do io
        println(io, "ic,traj,epoch,n_steps,loss,eta,pnorm,gnorm," * join(metric_keys, ","))
    end

    # Print information
    @info ".csv file created and initialized at $(filepath)!"

    return filepath
end


# Write a row of training data to .csv
function csv_row!(ic, traj, epoch, n_steps, loss, eta, pnorm, gnorm, metrics; path="", file="")

    # Put together filepath
    filepath = joinpath(path, file)

    # Write a data row
    open(filepath, "a") do io
        println(io, join((ic, traj, epoch, n_steps, loss, eta, pnorm, gnorm, values(metrics)...), ","))
    end

    return nothing
end


# Read .csv data for plotting
function csv_read(; path="", file="")

    # Put together filepath
    filepath = joinpath(path, file)

    # Read .csv data
    return CSV.read(filepath, DataFrame; comment="#")
end




# Print update after finishing one initial condition
function print_ic(ic, loss, pnorm, gnorm)
    println("\tIC $ic, Loss=$loss, |ps|=$pnorm, |g|=$gnorm")
end

# Print update after finishing one trajectory segment
function print_traj(traj, loss, pnorm, gnorm)
    println("\t\t\tTraj $traj, Loss=$loss, |ps|=$pnorm, |g|=$gnorm")
end



# Print information about the run configuration
function print_config(run_c, dt_sec)
    up_ic = run_c.n_traj * run_c.n_epochs
    up_total =  run_c.n_ic * up_ic

    t_gap = run_c.n_gap * dt_sec /3600

    n_steps_start = run_c.n_steps_0
    t_diff_start = n_steps_start * dt_sec /3600
    t_ic_start = run_c.n_ic * run_c.n_traj * (run_c.n_gap + n_steps_start) * dt_sec/ (3600*24)

    n_steps_end = run_c.n_steps_0 + run_c.n_steps_inc * (run_c.n_ic-1)
    t_diff_end = n_steps_end * dt_sec /3600
    t_ic_end = run_c.n_ic * run_c.n_traj * (run_c.n_gap + n_steps_end) * dt_sec/ (3600*24)

    println("\t\tNumber of updates per IC: ", up_ic)
    println("\t\tNumber of total updates: ", up_total)
    println("\t\tLength of gap timestepping (hours): ", t_gap)
    println("\t\tLength of trajectory differentiation (hours): Start: ", t_diff_start, "\tEnd: ", t_diff_end)
    println("\t\tTraining period per ic (days): Start: ", t_ic_start, "\tEnd: ", t_ic_end)
end