### Setup functions
###
### Functions for setting up and preparing simulations and optimisers for training



# Setup optimiser for online training loop
function setup_optimiser(tc; ps)

    # Copy learning rate
    eta = tc.eta0

    # Setup Optimiser rule
    rule = Optimisers.OptimiserChain(
        Optimisers.ClipNorm(1f0),           # ClipNorm for stability (limits gradients)
        Optimisers.WeightDecay(1f-2),       # WeightDecay for stability (limits parameters)
        Optimisers.Adam(eta),               # Adam optimiser with learning rate eta
    )

    # Define optimiser state
    opt_state = Optimisers.setup(rule, ps)

    return opt_state, eta
end



# Setup simulations for online training loop
function setup_simulations(spectral_grid, tc, lw_train)

    # Create template model and simulation for later copying
    model_template  = tc.model(spectral_grid; longwave_radiation = tc.lw_target)
    sim_template    = first_steps!(initialize!(model_template))

    # Create target simulation as training target
    model_target    = tc.model(spectral_grid; longwave_radiation = tc.lw_target)
    sim_target      = first_steps!(initialize!(model_target))

    # Create training simulation (to be trained scheme)
    model_train     = tc.model(spectral_grid; longwave_radiation = lw_train)
    sim_train       = first_steps!(initialize!(model_train))

    return (; template = sim_template, target = sim_target, train = sim_train,)
end



# Prepare reference simulation (copying, perturbation, spinup)
function prepare_reference(sim_template, tc, n_steps, start_date)

    # Copy template simulation for pertubation and set start date
    sim_pert = deepcopy(sim_template)
    SpeedyWeather.set!(sim_pert.variables.prognostic.clock; time=start_date, start=start_date)

    # Perturb temperature and humidity fields
    perturb_grid_field!(sim_pert, :temperature; fac_add = tc.fac_pert_T)
    perturb_grid_field!(sim_pert, :humidity; fac_mult = tc.fac_pert_q, zeromin = true)
        
    # Spinup simulation 
    run!(sim_pert, period = tc.t_spinup)

    # Create reference simulation
    sim_ref = deepcopy(sim_pert)

    # Initialize reference trajectory and do a first step
    steps = tc.n_traj * (tc.n_gap + n_steps) + 1
    first_steps!(sim_ref, planned_steps=steps)

    return sim_ref
end



# Samples starting dates uniformly across a year
function sample_start_date(ic, n_ic; start = DateTime(2000, 1, 1), rng = Random.default_rng())
    
    # Calculate bin size and sample a day within the bin
    bin = 365 /n_ic
    day = (ic-1)*bin + rand(rng)*bin

    # Return sampled day
    return start + Day(floor(Int, day))
end