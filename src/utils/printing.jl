### Printing utilities
###
### Helper functions for printing



# Print update after finishing one initial condition
function print_ic(ic, loss, pnorm, gnorm)
    println("\tIC $ic, Loss=$loss, |ps|=$pnorm, |g|=$gnorm")
end

# Print update after finishing one trajectory segment
function print_traj(traj, loss, pnorm, gnorm)
    println("\t\t\tTraj $traj, Loss=$loss, |ps|=$pnorm, |g|=$gnorm")
end

# Print update after a single optimization step
function print_epochs(epoch, loss, pnorm, gnorm)
    println("\t\t\t\t\tEpoch $epoch, Loss=$loss, |ps|=$pnorm, |g|=$gnorm")
end



# Print information about the run configuration
function print_config(run_c, dt_sec)
    up_ic = run_c.n_traj * run_c.n_epochs
    up_total =  run_c.n_ic * up_ic

    t_gap = run_c.n_gap * dt_sec /3600

    n_steps_start = run_c.n_steps_0
    t_diff_start = n_steps_start * dt_sec /3600
    t_ic_start = run_c.n_ic * run_c.n_traj * (run_c.n_gap + n_steps_start) * dt_sec/ (3600*24)

    n_steps_end = run_c.n_steps_0 + run_c.n_steps_inc * run_c.n_ic
    t_diff_end = n_steps_end * dt_sec /3600
    t_ic_end = run_c.n_ic * run_c.n_traj * (run_c.n_gap + n_steps_end) * dt_sec/ (3600*24)

    println("\t\tNumber of updates per IC: ", up_ic)
    println("\t\tNumber of total updates: ", up_total)
    println("\t\tLength of gap timestepping (hours): ", t_gap)
    println("\t\tLength of trajectory differentiation (hours): Start: ", t_diff_start, "\tEnd: ", t_diff_end)
    println("\t\tTraining period per ic (days): Start: ", t_ic_start, "\tEnd: ", t_ic_end)
end