### Generation of reference data sets
###
### Objective: Propagate a model with a target scheme and store its state for every simulated day
###     - full states are the restart points a rollout is initialized from (stored every full_gap days)
###     - grid and parameterization fields are the verification data (stored every day, much smaller)



# Generate a reference data set and store it as reference.jld2
function generate_reference(
    spectral_grid;
    name        = "default",                # name of the reference data set
    dir         = reference_dir(name),      # output directory
    seed        = 1234,                     # seed for RNG

    model       = PrimitiveWetModel,        # model used
    lw_scheme   = nothing,                  # longwave scheme the reference is generated with

    t_spinup    = Day(30),                  # spinup time before sampling
    start_date  = DateTime(2001, 1, 1),     # date of the first sampled state (day 0)
    sim_days    = 2*365,                    # number of sampled days
    full_gap    = 1,                        # store a full (restart) state every full_gap days

    fac_pert_T  = 2f0,                      # additive perturbation amplitude for temperature
    fac_pert_q  = 0.2f0,                    # multiplicative perturbation amplitude for humidity
)

    # Set seed for reproducability
    Random.seed!(seed)

    # Define model and initialize simulation
    sim = initialize!(model(spectral_grid; longwave_radiation = lw_scheme))

    # Set starting time so that the spinup ends at start_date
    clock_start = start_date - t_spinup
    SpeedyWeather.set!(sim.variables.prognostic.clock; time = clock_start, start = clock_start)

    # Perturb grid fields
    perturb_grid_field!(sim, :temperature; fac_add  = fac_pert_T)
    perturb_grid_field!(sim, :humidity;    fac_mult = fac_pert_q, zeromin = true)

    # Spinup simulation
    run!(sim, period = t_spinup)

    # Extract time step and calculate necessary steps for one day
    (; Δt) = sim.model.time_stepping
    steps_per_day = steps_from_days(1, Δt)

    # Initialize simulation (run! unscales vorticity/divergence) and perform a first timestep
    SpeedyWeather.initialize!(sim, steps = sim_days * steps_per_day + 1)
    SpeedyWeather.first_timesteps!(sim)


    ### Sample the simulation, one state per simulated day
    save_store(; dir, file = "reference.jld2") do store

        # Save the initial state (day 0), always a restart point
        store["day_0/full"]  = sim.variables
        store["day_0/grid"]  = sim.variables.grid
        store["day_0/param"] = sim.variables.parameterizations

        # Loop over the whole simulation
        for day in 1:sim_days

            # Propagate simulation for one day
            for _ in 1:steps_per_day
                SpeedyWeather.later_timestep!(sim)
            end

            # Save verification fields (small, every day)
            store["day_$(day)/grid"]  = sim.variables.grid
            store["day_$(day)/param"] = sim.variables.parameterizations

            # Save restart state (large, only every full_gap days)
            if day % full_gap == 0
                store["day_$(day)/full"] = sim.variables
            end
        end

        # Written last: their presence marks the file as complete
        store["full_gap"] = full_gap
        store["sim_days"] = sim_days
    end

    @info "Reference dataset $(name) stored at $(dir)!"

    return (; dir, sim_days, full_gap, steps_per_day)
end
