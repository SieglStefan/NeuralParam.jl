### Simulation functions
###
### Helper functions for running and starting simulations



# Calculate the number of timesteps from a number of days
steps_from_days(days, Δt_sec) = round(Int, days * 86400 / Δt_sec)

# Calculate the number of days from a number of timesteps
days_from_steps(n_steps, Δt_sec) = n_steps * Δt_sec / 86400



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

    # Copy (current) field for perturbation
    field = copy(SpeedyWeather.get_step(getfield(sim.variables.grid, var)))


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


    # Set variables onto the simulation and initialize again to apply perturbation
    SpeedyWeather.set!(sim; var => field)
    SpeedyWeather.initialize!(sim, steps=0)

    return nothing
end



# Initialize a simulation and do a first step (to initialize implicit solver)
function first_steps!(sim; steps=2)

    # Initialize simulation and do a first step
    SpeedyWeather.initialize!(sim, steps=steps)

    for _ in 1:2
        SpeedyWeather.time_step!(sim)
    end

    # Reinitialize simulation for continuation of time_step!() later
    SpeedyWeather.reinitialize!(sim.model, sim.variables)

    return sim
end



# Propagate a simulation for n_steps
function sim_timesteps!(sim, n_steps)

    # Propagate the simulation for n_steps
    for _ in 1:n_steps
        SpeedyWeather.time_step!(sim.variables, sim.model.time_stepping, sim.model)             # propagate dynamics
        SpeedyWeather.time_step!(sim.variables.prognostic.clock, sim.model.time_stepping)       # propagate clock
    end

    return nothing
end