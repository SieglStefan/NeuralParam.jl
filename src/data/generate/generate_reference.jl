### Generation of reference data sets
###
### Objective: Propagate a model with a target scheme and store its full state for every simulated day
###     - a full state serves both roles a rollout needs: it is the restart point the rollout is
###       initialized from, and the verification data it is scored against



# Generate a reference data set and store it as reference.jld2
function generate_reference(;
    spectral_grid,              # used spectral grid
    name,                       # name of the reference data set
    dir,                        # output directory
    seed,                       # seed for RNG

    model,                      # model used
    lw_scheme,                  # longwave scheme the reference is generated with

    t_spinup,                   # spinup time before sampling
    start_date,                 # date of the first sampled state (day 0)
    sim_days,                   # number of sampled days

    fac_pert_T,                 # additive perturbation amplitude for temperature
    fac_pert_q,                 # multiplicative perturbation amplitude for humidity
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

    
    # Initialize simulation (run! unscales vorticity/divergence) and perform the first steps
    first_steps!(sim; planned_steps = sim_days * steps_per_day + 1)


    ### Sample the simulation, one state per simulated day
    save_store(; dir, file = "reference.jld2") do store

        # Save the initial state (day 0)
        store["day_0"] = SpeedyWeather.materialize_views(sim.variables)

        # Loop over the whole simulation
        for day in 1:sim_days

            # Propagate simulation for one day
            sim_timesteps!(sim, steps_per_day)

            # Save the full state — restart point and verification data in one
            store["day_$(day)"] = SpeedyWeather.materialize_views(sim.variables)
        end

        # Written last: their presence marks the file as complete
        store["steps_per_day"] = steps_per_day
        store["sim_days"]      = sim_days

    end

    @info "Reference dataset $(name) stored at $(dir)!"

    return (; dir, steps_per_day, sim_days)
end
