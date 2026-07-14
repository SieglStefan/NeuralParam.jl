### Data utilities
###
### Helper functions for handling data



# Function for perturbing a grid variable field of a simulation
function perturb_grid_field!(
    sim,
    var::Symbol; 
    fac_add = 0f0,
    fac_mult = 0f0,
    offset = 0f0,
    zeromin = false,
    rng = Random.default_rng()
)
    
    if !hasfield(typeof(sim.variables.grid), var)
        @warn "Field $var does not exist in used model — perturbation skipped!." maxlog=1
        return nothing
    end

    # Initalize simulation (fill variables.grid if not initialized yet)
    SpeedyWeather.initialize!(sim, steps=0)

    # Copy field for perturbation
    field = copy(getfield(sim.variables.grid, var))


    # Additive perturbation
    field .+= fac_add .* randn!(rng, similar(field))

    # Multiplicative perturbation
    field .*= 1f0 .+ fac_mult .* randn!(rng, similar(field))

    # Offset
    field .+= offset

    # Only take positive values if set
    if zeromin
        field .= max.(field, 0f0)
    end


    # Set variables onto the simulation and initialize again to apply
    SpeedyWeather.set!(sim; var => field)
    SpeedyWeather.initialize!(sim, steps=0)

    return nothing
end




# Propagate a simulation for n_steps using a leapfrog timestep!()
function sim_timesteps!(sim, n_steps)

    # Extract time stepping
    dt = 2 * sim.model.time_stepping.Δt

    # Propagate the simulation for n_steps * dt
    for _ in 1:n_steps
        SpeedyWeather.timestep!(sim.variables, dt, sim.model, 2, 2)
    end

    return nothing
end



# Function for sampling a trajectory of sim, starting with ic
function sample_trajectory(sim, ic; n_steps, n_gap)
    
    # Initialize sim, do a first step and set initial condition
    SpeedyWeather.initialize!(sim; steps = n_steps)
    SpeedyWeather.first_timesteps!(sim)
    copy!(sim.variables, ic)

    # Create container for grid temperatures with a first entry
    data = [copy(sim.variables.grid.temperature)]

    # Loop over steps
    for step in 1:n_steps

        # Do a later  timestep
        SpeedyWeather.later_timestep!(sim)

        # Store temperature after n_gaps
        step % n_gap == 0 && push!(data, copy(sim.variables.grid.temperature))
    end

    return (; temperature = data)
end