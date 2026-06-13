





    

# Generate a series of temperature fields from a model
function generate_temperature_fields(
    model_base,             # base model used for spinup
    model_data;             # data model used for generating temperature fields
    t_spinup = 14,          # spinup time in days
    t_sim = 1,              # simulation time in days
    save_every = 1,         # save every n timesteps
)

    # Copy models
    model_base_copy = deepcopy(model_base)
    model_data_copy = deepcopy(model_data)

    # Create and spin up base simulation
    sim_base = SpeedyWeather.initialize!(model_base_copy)
    run!(sim_base, period = Day(t_spinup))

    # Create data simulation from spun-up base state
    sim_data = SpeedyWeather.initialize!(model_data_copy)
    copy!(sim_data.variables, sim_base.variables)


    # Extract timestep size
    Δt_sec = model_data_copy.time_stepping.Δt_sec

    # Compute number of timesteps and saved datapoints
    n_steps = Int(round(t_sim * 24 * 3600 / Δt_sec))
    n_data = Int(floor(n_steps / save_every))

    @info "Generated $n_data datapoints with temporal distance of $(save_every * Δt_sec) seconds!"


    # Container for saved temperature fields
    T = Vector{typeof(sim_data.variables.grid.temperature)}()

    # Initialize simulation and perform first startup timestep
    SpeedyWeather.initialize!(sim_data, steps = n_steps + 1)
    SpeedyWeather.first_timesteps!(sim_data)

    
    # Run simulation and save temperature fields
    for step in 1:n_steps
        SpeedyWeather.timestep!(sim_data)

        if step % save_every == 0
            push!(T, copy(sim_data.variables.grid.temperature))
        end
    end

    return T
end



# Perturb the grid temperature field of a simulation with additive white noise
function perturb_grid_temp!(sim; amp = 2.0, rng = Random.default_rng())

    # Initialize simulation if needed; otherwise grid variables may be empty
    initialize!(sim)

    # Copy grid temperature field and add white noise
    T_grid = copy(sim.variables.grid.temperature)
    noise = randn!(rng, similar(T_grid))

    T_grid .+= amp .* noise

    # Use set! so that prognostic variables are updated consistently
    set!(sim, temperature = T_grid)
    initialize!(sim)

    return nothing
end


# Perturb the grid humidity field of a simulation with multiplicative white noise
function perturb_grid_humid!(sim; amp = 0.1f0, rng = Random.default_rng())

    # Initialize simulation if needed; otherwise grid variables may be empty
    initialize!(sim)

    # Copy grid humidity field and apply relative perturbation
    q_grid = copy(sim.variables.grid.humidity)
    noise = randn!(rng, similar(q_grid))

    q_grid .*= 1f0 .+ amp .* noise
    q_grid .= max.(q_grid, 0f0)

    # Use set! so that prognostic variables are updated consistently
    set!(sim, humidity = q_grid)
    initialize!(sim)

    return nothing
end