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
    
    # Check if grid has variable var
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



# Function for sampling a trajectory of sim, starting with initial_condition
function sample_grid_trajectory(
    sim,
    vars, 
    initial_condition; 
    n_steps, n_gap
)

    # Check if grid has variables vars
    for v in vars
        if !hasfield(typeof(sim.variables.grid), v)
            @warn "Field $v does not exist in used model - trajectory sampling aborted!" maxlog=1
            return nothing
        end
    end

    # Initialize sim, do a first step and set initial condition
    SpeedyWeather.initialize!(sim; steps = n_steps+1)
    SpeedyWeather.first_timesteps!(sim)
    copy!(sim.variables, initial_condition)

    # Create container for grid fields with a first entry
    data = (; (v => [copy(getfield(sim.variables.grid, v))] for v in vars)...)

    # Loop over steps
    for step in 1:n_steps

        # Do a later timestep
        SpeedyWeather.later_timestep!(sim)

        # Store temperature after n_gaps
        
        if step % n_gap == 0
            for v in vars
                push!(data[v], copy(getfield(sim.variables.grid, v)))
            end
        end
    end

    return data
end


sample_grid_trajectory(sim, var::Symbol, initial_condition; n_steps, n_gap) = 
    sample_grid_trajectory(sim, (var,), initial_condition; n_steps=n_steps, n_gap=n_gap)