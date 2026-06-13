### Printing utilities for optimization progress
###
### Small helper functions for printing loss, parameter values,
### and parameter/gradient norms during optimization.



# Print update after finishing one initial condition
function print_ic(ic, loss)
    println("IC $ic, Loss=$loss")
end


# Print update after finishing one trajectory segment
function print_traj(traj, loss)
    println("\t\tTrajectory $traj, Loss=$loss")
end


# Print update after a single optimization step for ConstLinearLongwave
function print_epochs(radiation::AbstractConstLongwave, epoch, loss, grads_opt)
    println(
        "\t\t\tEpoch $epoch, Loss=$loss, " *
        "a=$(radiation.a), b=$(radiation.b), " *
        "ga_sc=$(grads_opt.a), gb_sc=$(grads_opt.b)"
    )
end


# Print update after a single optimization step for neural LinearLongwave variants
function print_epochs(radiation::AbstractLuxLongwave, epoch, loss, grads_opt)
    pnorm = tree_l2norm(radiation.ps)
    gnorm = tree_l2norm(grads_opt)

    println("\t\t\tEpoch $epoch, Loss=$loss, |ps|=$pnorm, |g|=$gnorm")
end