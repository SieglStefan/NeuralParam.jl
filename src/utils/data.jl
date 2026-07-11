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



function sample_sims(spectral_grid, schemes::NamedTuple; fac_pert_T, fac_pert_q, t_spinup, sim_time, sample_gap)
    sim_pert = initialize!(PrimitiveWetModel(spectral_grid))
    (; Δt_sec) = sim_pert.model.time_stepping
    n_steps = round(Int, sim_time   * 86400 / Δt_sec) + 1
    n_gap   = round(Int, sample_gap * 86400 / Δt_sec)

    perturb_grid_field!(sim_pert, :temperature; fac_add = fac_pert_T)
    perturb_grid_field!(sim_pert, :humidity;    fac_mult = fac_pert_q, zeromin = true)
    run!(sim_pert, period = t_spinup)

    ic = deepcopy(sim_pert.variables)                     # geteilte Anfangsbedingung

    trajectories = map(schemes) do scheme
        model = PrimitiveWetModel(spectral_grid; longwave_radiation = scheme)   # FRISCH — jeder Typ ok, auch nothing
        sim   = initialize!(model)
        sample_trajectory(sim, ic; n_steps, n_gap)
    end

    return (; trajectories, Δt_sample = n_gap * Δt_sec)
end




function sample_trajectory(sim, ic; n_steps, n_gap)
    SpeedyWeather.initialize!(sim; steps = n_steps)
    SpeedyWeather.first_timesteps!(sim)
    copy!(sim.variables, ic)                              # geteilte IC NACH dem Setup → bleibt erhalten (behebt die init-Reihenfolge)

    data = [copy(sim.variables.grid.temperature)]
    for step in 1:n_steps
        SpeedyWeather.later_timestep!(sim)
        step % n_gap == 0 && push!(data, copy(sim.variables.grid.temperature))
    end
    return (; temperature = data)
end
