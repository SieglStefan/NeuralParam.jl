### Utility functions for pertubing ICs of simulations



# Function for pertubing the grid temperature field of a simulation for IC sampling using white noise
function perturb_grid_temp!(sim; amp=2., rng=Random.default_rng())

    # Initialize simulation (if not initialized yet, T_grid is empty)
    initialize!(sim)

    # Copy the grid temperature field
    T_grid = copy(sim.variables.grid.temperature)

    # Create white noise
    noise = randn!(rng, similar(T_grid))

    # Add noise to the grid temperature field
    T_grid .+= amp .* noise

    # Assigning the perturbated field to the simulation and initialize it (without initialize!, sim.prognostic is not actualized)
    set!(sim, temperature = T_grid)
    initialize!(sim)

    return nothing
end