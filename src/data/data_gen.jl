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



# Samples starting dates uniformly across a year
function sample_start_date(ic, n_ic; year=2000)
    
    # Calculate bin size and sample a day within the bin
    bin = 365 /n_ic
    day = (ic-1)*bin + rand()*bin

    # Return sampled day
    return DateTime(year, 1, 1) + Day(floor(Int, day))
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



# Default probes for verification: the fields a longwave scheme is judged on
#   - a probe reads one field out of a state, so the same probe works on a running simulation
#     (sim.variables) and on a stored reference state (verify_state)
const PROBES = (;
    T    = v -> SpeedyWeather.get_step(v.grid.temperature),
    olw  = v -> v.parameterizations.outgoing_longwave,
    slwd = v -> v.parameterizations.surface_longwave_down,
    slwu = v -> v.parameterizations.surface_longwave_up,
)


# Function for sampling probes along a trajectory of sim, starting with initial_condition
#   - propagates n_samples * n_gap timesteps and returns n_samples + 1 fields per probe,
#     the first one being the initial condition itself
function sample_trajectory(
    sim,
    probes,
    initial_condition;
    n_samples, n_gap
)

    # Initialize sim, do a first step and set initial condition
    first_steps!(sim; steps = n_samples * n_gap + 1)
    copy!(sim.variables, initial_condition)

    # Create container for probed fields with a first entry
    data = (; (p => [copy(probe(sim.variables))] for (p, probe) in pairs(probes))...)


    # Loop over samples
    for _ in 1:n_samples

        # Do n_gap steps
        sim_timesteps!(sim, n_gap)

        # Store probed fields
        for (p, probe) in pairs(probes)
            push!(data[p], copy(probe(sim.variables)))
        end
    end

    return data
end



# Function for sampling vars of a trajectory of sim, starting with initial_condition
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
    first_steps!(sim; steps=n_steps+1)
    copy!(sim.variables, initial_condition)

    # Create container for grid fields with a first entry
    data = (; (v => [copy(SpeedyWeather.get_step(getfield(sim.variables.grid, v)))] for v in vars)...)


    # Loop over steps
    for step in 1:n_steps

        # Do n_gap steps
        sim_timesteps!(sim, n_gap)

        # Store variables
        for v in vars
            push!(data[v], copy(SpeedyWeather.get_step(getfield(sim.variables.grid, v))))
        end
    end

    return data
end

# Wrapper for a single var symbol (allowing a single variable instead of tuple)
sample_grid_trajectory(sim, var::Symbol, initial_condition; n_steps, n_gap) = 
    sample_grid_trajectory(sim, (var,), initial_condition; n_steps=n_steps, n_gap=n_gap)