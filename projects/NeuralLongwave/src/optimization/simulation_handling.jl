### Simulation handling utilities
###
### Helper functions for creating, resetting, and propagating simulations
### during online optimization.



# Create a template simulation that can be copied for faster setup
function create_template(spectral_grid)

    # Create template model and simulation
    model_template = PrimitiveWetModel(; spectral_grid)
    sim_template = initialize!(model_template)

    return sim_template
end



# Create reference, target, and train simulations from a perturbed and spun-up template state
function create_sims(spectral_grid, sim_template; radiation, t_spinup, amp_pert_T, amp_pert_q)
    
    # Copy template simulation
    sim_pert = deepcopy(sim_template)

    # Perturb temperature and humidity fields and spin up to obtain a random IC
    perturb_grid_temp!(sim_pert, amp = amp_pert_T)
    perturb_grid_humid!(sim_pert, amp = amp_pert_q)
    run!(sim_pert, period = Hour(t_spinup))

    # Reference and target simulations start from the same perturbed state
    sim_ref = deepcopy(sim_pert)
    sim_target = deepcopy(sim_pert)

    # Training simulation uses the optimized longwave parameterization
    model_train = PrimitiveWetModel(; spectral_grid, longwave_radiation = radiation)
    sim_train = initialize!(model_train)
    copy!(sim_train.variables, sim_pert.variables)

    return sim_ref, sim_target, sim_train
end



# Reset a simulation to a stored Variables state
function reset_sim!(sim, vars0)
    
    copy!(sim.variables, vars0)

    return nothing
end



# Prepare target/train simulations for one AD trajectory segment
function prepare_sim_pair!(sim_target, sim_train, n_steps)
    
    # Initialize simulations with the correct number of steps
    SpeedyWeather.initialize!(sim_target, steps = n_steps + 1)
    SpeedyWeather.initialize!(sim_train, steps = n_steps + 1)

    # Perform the first timestep/startup step
    SpeedyWeather.first_timesteps!(sim_target)
    SpeedyWeather.first_timesteps!(sim_train)

    # Store initial variables before the differentiated trajectory
    vars0 = deepcopy(sim_target.variables)

    return vars0
end



# Propagate a simulation for n_steps using the leapfrog timestep size
function sim_timesteps!(sim, n_steps)

    # Extract time stepping
    dt = 2 * sim.model.time_stepping.Δt

    # Propagate the simulation for n_steps * dt
    for _ in 1:n_steps
        SpeedyWeather.timestep!(sim.variables, dt, sim.model)
    end

    return nothing
end