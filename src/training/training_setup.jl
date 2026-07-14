
















function setup_simulations(
    spectral_grid,
    run_config,
    lw_radiation_train,
    lw_radiation_target,
)

    # Create template model and simulationfor later copying
    model_template = run_config.model_type(; spectral_grid)
    sim_template = initialize!(model_template)

    # Create target and training simulation and do a first timestep (initalize implicit solver)
    model_target = run_config.model_type(; spectral_grid, longwave_radiation = lw_radiation_target)
    sim_target   = initialize!(model_target)
    SpeedyWeather.initialize!(sim_target, steps=0)
    SpeedyWeather.first_timesteps!(sim_target)

    model_train  = run_config.model_type(; spectral_grid, longwave_radiation = lw_radiation_train)
    sim_train    = initialize!(model_train)
    SpeedyWeather.initialize!(sim_train, steps=0)
    SpeedyWeather.first_timesteps!(sim_train)

    return sim_template, sim_train, sim_target
end



function setup_optimiser(
    run_config;
    ps
)
    eta = run_config.eta0

    rule = Optimisers.OptimiserChain(
        Optimisers.ClipNorm(1f0),
        Optimisers.WeightDecay(1f-4),
        Optimisers.Adam(eta),
    )

    opt_state = Optimisers.setup(rule, ps)

    return opt_state, eta
end



function prepare_reference(sim_template, run_config, n_steps, start_date)

    # Copy template simulation for pertubation and set start date
    sim_pert = deepcopy(sim_template)
    SpeedyWeather.set!(sim_pert.variables.prognostic.clock; time=start_date, start=start_date)

    # Perturb temperature and humidity fields
    perturb_grid_field!(sim_pert, :temperature; fac_add = run_config.fac_pert_T)
    perturb_grid_field!(sim_pert, :humidity; fac_mult = run_config.fac_pert_q, zeromin = true)
        
    # Spinup simulation 
    run!(sim_pert, period = Hour(run_config.t_spinup))

    # Create reference simulation
    sim_ref = deepcopy(sim_pert)

    # Initialize reference trajectory and do a first step
    SpeedyWeather.initialize!(sim_ref, steps = run_config.n_traj * (run_config.n_gap + n_steps) + 1)
    SpeedyWeather.first_timesteps!(sim_ref)

    return sim_ref
end
