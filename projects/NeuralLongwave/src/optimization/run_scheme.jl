### Wrapper functions for running calibration and training
###
### These functions provide simple entry points for optimizing longwave
### parameterizations. Both wrappers call the same generic optimization loop.



# Convenience wrapper for calibrating a constant LinearLongwave parameterization
function run_calibration!(radiation::AbstractConstLongwave, spectral_grid; kwargs...)

    return run_optimization!(;
        radiation,
        spectral_grid,
        kwargs...,
    )
end


# Convenience wrapper for training a NeuralLinearLongwave parameterization
function run_training!(radiation::AbstractLuxLongwave, spectral_grid; kwargs...)
    
    return run_optimization!(;
        radiation,
        spectral_grid,
        kwargs...,
    )
end



# Generic optimization wrapper used by run_calibration! and run_training!()
function run_optimization!(;
    radiation,                   # parameterization to be optimized
    spectral_grid,               # spectral grid used for model construction

    eta0 = 1f-3,                 # initial learning rate
    eta_fac = 1f0,               # learning-rate decay factor
    eta_steps = 10,              # apply eta_fac every eta_steps

    t_spinup = Day(14),          # spinup time before IC sampling

    n_ic = 10,                   # number of initial conditions
    n_traj = 100,                # number of trajectories per IC
    n_epochs = 50,               # number of updates per trajectory
    n_gap = 10,                  # timesteps between calibration points
    n_steps = 1,                 # timesteps used in autodiff/checkpointing

    printing_ic = true,          # print after every IC
    printing_traj = true,        # print after every trajectory update
    printing_epochs = false,     # print after every epoch

    test_mode = false,           # skip Enzyme.autodiff if true
)

    # Containers for logging
    L = Float32[]       # loss
    P = []              # full parameters, e.g. (a,b) or ps
    G = []              # full gradients with respect to P
    PN = Float32[]      # parameter norm
    GN = Float32[]      # gradient norm


    # Possible: run first offline optimization to get good initial guess for online optimization


    # Warn when running without autodiff
    if test_mode
        @warn "Test mode is activated! Enzyme.autodiff is NOT used!"
    end


    # Run online optimization loop
    online_optimization!(;
        radiation,
        spectral_grid,
        eta0,
        eta_fac,
        eta_steps,
        t_spinup,
        n_ic,
        n_traj,
        n_epochs,
        n_gap,
        n_steps,
        printing_ic,
        printing_traj,
        printing_epochs,
        test_mode,
        L, P, G, PN, GN
    )

    return L, P, G, PN, GN
end