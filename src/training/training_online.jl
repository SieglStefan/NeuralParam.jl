### Trains a parameterization online in SpeedyWeather,jl
###
### Algorithm:
###     
###     - Initialize containers, optimiser and training data saving
###     - Initialize target and training simulation:        sim_target and sim_train
###     - Create a template simulation for copying:         sim_template
###     
###     - Loop over initial conditions: n_ic
###         - Copy sim_template and perturb it:             sim_pert
###         - Spinup sim_pert
###         - Copy sim_pert for reference:                  sim_ref
###
###         - Loop over trajectories: n_traj
###             - Copy sim_ref onto sim_target and sim_train
###             - Propagate sim_target for n_steps steps
###
###             - Loop over epochs: n_epochs
###                 - Update radiation scheme parameters
###                 - Propagate sim_train for n_steps
###                 - Calculate gradients
###                 - Store training data
###
###             - Propagate sim_ref for n_gap steps
###
###         - Update learning rate



# Function for running a online training
function training_online(;
    spectral_grid,
    lw_radiation_train,
    lw_radiation_target,
    run_config,
    output_config,
    output_path,
    test_mode,
)

    # Unpack run config parameters
    (; eta_decay, patience, min_delta,
    n_ic, n_traj, n_epochs, n_steps_0, n_steps_inc, n_gap,
    ) = run_config


    # Set seed for reproducability
    Random.seed!(run_config.seed)


    # Setup optimiser
    opt_state, eta = setup_optimiser(run_config, ps=lw_radiation_train.ps)
   
    # Define variables for training control
    best_loss = Inf32
    stale = 0
    best_ps = deepcopy(ps)


    # Setup simulations
    sim_template, sim_train, sim_target = setup_simulations(
        spectral_grid,
        lw_radiation_train,
        lw_radiation_target,
    )


    # Define containers for logging
    L = Float32[]       # loss
    PN = Float32[]      # parameter norm
    GN = Float32[]      # gradient norm


    # Initalize .csv file for logging
    if output_config.train_save
        meta = build_meta(lw_radiation_train, lw_radiation_target, run_config)
        metric_keys = keys(compute_metrics(lw_radiation_train, sim_train.variables, sim_target.variables))
        csv_init(meta, metric_keys; path=output_path, file=output_config.train_file)
    end


    # Print training start information
    @info "Online training started!"
    print_config(run_config, 2*sim_template.model.time_stepping.Δt_sec)

    if test_mode
        @warn "Test mode is activated! Enzyme.autodiff is NOT used!"
    end



    # Loop over initial conditions
    for ic in 1:n_ic
    
        # Update number of steps for calculating gradients
        n_steps = n_steps_0 + (ic-1) * n_steps_inc

        # Draw a starting date
        start_date = sample_start_date(ic, n_ic)
    
        # Prepare reference simulation 
        sim_ref = prepare_reference(sim_template, run_config, n_steps, start_date)



        # Loop over trajectory segments
        for traj in 1:n_traj

            # Copy reference variables
            vars0 = deepcopy(sim_ref.variables) 


            ### Prepare target simulation

            # Set target variables to reference variables
            copy!(sim_target.variables, vars0)

            # Propagate target simulation for gradient computation
            sim_timesteps!(sim_target, n_steps)



            # Reuse same trajectory segment for several updates
            for epoch in 1:n_epochs

                ### Prepare training simulation
            
                # Re-create training simulation with updated radiation
                sim_train = @set sim_train.model.longwave_radiation = lw_radiation_train

                # Set training variables to reference variables
                copy!(sim_train.variables, vars0)

                # Propagate training simulation for gradient computation
                sim_timesteps!(sim_train, n_steps)


                # Print information of starting first training step
                if ic == 1 && traj== 1 && epoch == 1
                    @info "Start 1st training step!"
                end


                # Perform one training step
                step = online_training_step(
                    lw_radiation_train;
                    vars0,
                    sim_target,
                    sim_train,
                    n_steps,
                    opt_state,
                    test_mode
                )            
                
                lw_radiation_train = step.lw_radiation
                opt_state = step.opt_state


                # Store loss, parameters, gradients, and norms
                push!(L, step.loss)
                push!(PN, Float32(tree_l2norm(step.lw_radiation.ps)))
                push!(GN, Float32(tree_l2norm(step.grads)))

                # Write to .csv
                if output_config.train_save
                    csv_row!(
                        ic, traj, epoch, n_steps, 
                        L[end], eta, PN[end], GN[end], step.metrics;
                        path=output_path, file=output_config.train_file
                    )
                end
                    

                # Print epoch update
                if output_config.printing_epochs
                    print_epochs(epoch, L[end], PN[end], GN[end])
                end
            end


            # Propagate reference trajectory forward
            for _ in 1:(n_steps+n_gap)
                SpeedyWeather.later_timestep!(sim_ref)
            end
            
            
            # Print trajectory update
            if output_config.printing_traj
                print_traj(traj, L[end], PN[end], GN[end])
            end
        end


        # Update learning rate
        eta *= eta_decay
        Optimisers.adjust!(opt_state, eta)

        loss_ic_mean = mean(L[end-n_traj*n_epochs+1:end])

        if loss_ic_mean < best_loss - min_delta
            best_loss = loss_ic_mean
            best_ps = deepcopy(lw_radiation_train.ps)
            stale = 0
        else
            stale += 1
            if stale >= patience 
                @warn "Training finished early! (no loss decrease)"
                break
            end
        end


        # Print IC update
        if output_config.printing_ic
            print_ic(ic, L[end], PN[end], GN[end])
        end

        # Plot current loss trajectory after every ic
        if output_config.live_plots
            display(plot_training(L, PN, GN))
        end
    end 
    
    @info "Training finished!"

    return update_ps(lw_radiation_train, best_ps), L, PN, GN
end



# Function for performing one training step
function online_training_step(
    lw_radiation_train;
    vars0,
    sim_target,
    sim_train,
    n_steps,
    opt_state,
    test_mode,
)

    # Compute gradients 
    grads, loss, metrics = compute_gradients(
        vars0, 
        sim_target, 
        sim_train, 
        n_steps, 
        test_mode
    )

    # Update parameters
    opt_state, ps_new = Optimisers.update(opt_state, lw_radiation_train.ps, grads)
    lw_radiation_new = update_ps(lw_radiation_train, ps_new)

    return (; lw_radiation=lw_radiation_new, loss, metrics, grads, opt_state)
end



# Helper function for updating parameterization parameters
function update_ps(lw::ConstLinearLW, ps_new)
    return ConstLinearLW(lw.scaling, ps_new)
end

# Helper function for updating parameterization parameters
function update_ps(lw::NeuralLinearLW, ps_new)
    return NeuralLinearLW(lw.n_in, lw.n_out, lw.arch_config, lw.zscore, lw.scaling, lw.nn, ps_new, lw.st, lw.input_buffer)
end

# Helper function for updating parameterization parameters
function update_ps(lw::NeuralABRLW, ps_new)
    return NeuralABRLW(lw.n_in, lw.n_out, lw.arch_config, lw.zscore, lw.nn, ps_new, lw.st, lw.input_buffer, lw.def_co2, lw.def_ocean_em, lw.def_land_em)
end

# Helper function for updating parameterization parameters
function update_ps(lw::NeuralABRLWGlobal, ps_new)
    return NeuralABRLWGlobal(lw.n_in, lw.n_out, lw.n_points, lw.arch_config, lw.zscore, lw.nn, ps_new, lw.st, lw.def_co2, lw.def_ocean_em, lw.def_land_em)
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



function setup_simulations(
    spectral_grid,
    lw_radiation_train,
    lw_radiation_target,
)

    # Create template model and simulationfor later copying
    model_template = PrimitiveWetModel(; spectral_grid)
    sim_template = initialize!(model_template)

    # Create target and training simulation and do a first timestep (initalize implicit solver)
    model_target = isnothing(lw_radiation_target) ?
        PrimitiveWetModel(; spectral_grid) :
        PrimitiveWetModel(; spectral_grid, longwave_radiation = lw_radiation_target)
    sim_target   = initialize!(model_target)
    SpeedyWeather.initialize!(sim_target, steps=0)
    SpeedyWeather.first_timesteps!(sim_target)

    model_train  = PrimitiveWetModel(; spectral_grid, longwave_radiation = lw_radiation_train)
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

    rule = Optimisers.Adam(eta)
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



# XXX
function sample_start_date(ic, n_ic; year=2000)
    bin = 365 / n_ic
    doy = (ic-1)*bin + rand()*bin
    return DateTime(year, 1, 1) + Day(floor(Int, doy))
end