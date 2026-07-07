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
    up_total =  run_c.n_ic * run_c.n_traj * run_c.n_epochs

    n_steps_start = run_c.n_steps_0
    t_ic_start = run_c.n_ic * run_c.n_traj * (run_c.n_gap + n_steps_start) * dt_sec/ (3600*24)

    n_steps_end = run_c.n_steps_0 + run_c.n_steps_inc * run_c.n_ic
    t_ic_end = run_c.n_ic * run_c.n_traj * (run_c.n_gap + n_steps_end) * dt_sec/ (3600*24)

    @info "Number of total updates: " up_total
    @info "Training period per ic start (days): " t_ic_start
    @info "Training period per ic end (days): " t_ic_end
end