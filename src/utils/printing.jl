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
    t_total = run_c.n_ic * run_c.n_traj * (run_c.n_gap + run_c.n_steps) * dt_sec/ (3600*24)

    @info "Number of total updates: " up_total
    @info "Training period (days): " t_total
end