### Printing utilities for optimization progress
###
### Small helper functions for printing loss, parameter values,
### and parameter/gradient norms during optimization.



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



function print_config(c, dt_sec)
    up_total =  c.n_ic * c.n_traj * c.n_epochs
    t_total = c.n_ic * c.n_traj * (c.n_gap + c.n_steps) * dt_sec/ (3600*24)

    @info "Number of total updates: " up_total
    @info "Training period (days): " t_total
end


