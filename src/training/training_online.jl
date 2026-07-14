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
    (; eta_decay,
    n_ic, n_traj, n_epochs, n_steps_0, n_steps_inc, n_gap,
    ) = run_config


    # Set seed for reproducability
    Random.seed!(run_config.seed)


    # Setup optimiser
    opt_state, eta = setup_optimiser(run_config, ps=lw_radiation_train.ps)


    # Setup simulations
    sim_template, sim_train, sim_target = setup_simulations(
        spectral_grid,
        run_config,
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

    # Initialize folder for training plots
    if output_config.plots_save
        dir = joinpath(output_path, output_config.plots_folder)
        mkpath(dir)
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


        # Print IC update
        if output_config.printing_ic
            print_ic(ic, L[end], PN[end], GN[end])
        end


        # Plot current loss trajectory after every ic
        if output_config.plots_save
            p = plot_training(L, PN, GN;
                plot_kwargs=(;plot_title = "until IC nr. $(ic)"))

            dir = joinpath(output_path, output_config.plots_folder)
            file = joinpath(dir, "IC_$(ic).png")
            
            Plots.savefig(p, file)
        end
    end 
    
    @info "Training finished!"

    return lw_radiation_train, L, PN, GN
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



