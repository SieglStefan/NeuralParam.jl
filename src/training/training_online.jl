### Trains a parameterization online in SpeedyWeather.jl
###
### XXX Algorithm:
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
    run_config,
    output_config,
    output_path,
)

    # Set seed for reproducability
    Random.seed!(run_config.seed)


    # Setup optimiser
    opt_state, eta = setup_optimiser(run_config, ps=lw_radiation_train.ps)

    # Setup simulations
    sim_template, sim_train, sim_target = setup_simulations(
        spectral_grid,
        run_config,
        lw_radiation_train,
    )


    # Calculate grid weights for loss function
    grid_weights = area_weights(sim_train.variables.grid.temperature)


    # Define containers for logging
    L = Float32[]       # loss
    PN = Float32[]      # parameter norm
    GN = Float32[]      # gradient norm


    # Initalize .csv file for logging
    if output_config.train_save
        metric_keys = keys(compute_metrics(lw_radiation_train, sim_train.variables, sim_target.variables))
        csv_init(metric_keys; path=output_path, file=output_config.train_file)
    end

    # Initialize folder for training plots
    if output_config.plots_save
        dir = joinpath(output_path, output_config.plots_folder)
        mkpath(dir)
    end


    # Print training start information
    @info "Online training started!"
    print_config(run_config, 2*sim_template.model.time_stepping.Δt_sec)

    if ~run_config.do_autodiff
        @warn "Autodiff is deactivated! Enzyme.autodiff is NOT used!"
    end


    # Shuffle ics
    bin_order = randperm(run_config.n_ic)

    # Loop over initial conditions
    for ic in 1:run_config.n_ic
    
        # Update number of steps for calculating gradients
        n_steps = run_config.n_steps_0 + (ic-1) * run_config.n_steps_inc

        # Draw a starting date
        start_date = sample_start_date(bin_order[ic], run_config.n_ic)
    
        # Prepare reference simulation 
        sim_ref = prepare_reference(sim_template, run_config, n_steps, start_date)


        # Declare gradient sum and update flag for training step
        grad_sum = nothing
        do_update = false



        # Loop over trajectory segments
        for traj in 1:run_config.n_traj

            # Copy reference variables
            vars0 = deepcopy(sim_ref.variables) 


            # Set target variables to reference variables
            copy!(sim_target.variables, vars0)

            # Propagate target simulation for gradient computation
            sim_timesteps!(sim_target, n_steps)



            # Reuse same trajectory segment for several updates
            for epoch in 1:run_config.n_epochs

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

                # Update flag for training step
                if traj % run_config.n_batch == 0
                    do_update = true
                end


                # Perform one online training step
                step = online_training_step(;
                    lw_radiation_train,
                    do_update,
                    grad_sum,
                    vars0,
                    sim_target,
                    sim_train,
                    n_steps,
                    opt_state,
                    n_batch = run_config.n_batch,
                    do_autodiff = run_config.do_autodiff,
                    grid_weights
                )            


                # Extract gradient sum and optimser state
                grad_sum = step.grad_sum
                opt_state = step.opt_state

                # Update radiation scheme parameters after batch window
                if do_update
                    lw_radiation_train = step.lw_radiation
                    do_update = false
                end


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
            end


            # Propagate reference trajectory forward
            for _ in 1:(n_steps+run_config.n_gap)
                SpeedyWeather.later_timestep!(sim_ref)
            end
            
            
            # Print trajectory update
            if output_config.printing_traj
                print_traj(traj, L[end], PN[end], GN[end])
            end
        end


        # Update learning rate after every ic and update optimiser
        eta *= run_config.eta_decay
        Optimisers.adjust!(opt_state, eta)


        # Print IC update
        if output_config.printing_ic
            print_ic(ic, L[end], PN[end], GN[end])
        end


        # Plot current loss trajectory after every ic
        if output_config.plots_save
            p = plot_training(L, PN, GN;
                plot_kwargs=(;plot_title = "until IC nr. $(ic) / $(run_config.n_ic)"),)

            dir = joinpath(output_path, output_config.plots_folder)
            file = joinpath(dir, "IC_$(ic).png")
            
            Plots.savefig(p, file)
        end
    end 
    
    @info "Training finished!"

    return lw_radiation_train, L, PN, GN
end



# Function for performing one training step
function online_training_step(;
    lw_radiation_train,
    do_update,
    grad_sum,
    vars0,
    sim_target,
    sim_train,
    n_steps,
    opt_state,
    n_batch,
    do_autodiff,
    grid_weights,
)

    # Compute gradients and rest
    grads, loss, metrics = compute_gradients(
        vars0, 
        sim_target, 
        sim_train, 
        n_steps, 
        do_autodiff,
        grid_weights
    )

    # Accumulate gradients over batch window
    grad_sum = isnothing(grad_sum) ? grads : tree_add(grad_sum, grads)


    # Calculate mean gradients when scheme update is due
    if do_update

        # Calculate mean
        avg = tree_scale(grad_sum, 1f0/n_batch)

        # Update optimiser and scheme
        opt_state, ps_new = Optimisers.update(opt_state, lw_radiation_train.ps, avg)
        lw_radiation_train = update_ps(lw_radiation_train, ps_new)

        # Reset gradient sum
        grad_sum = nothing
    end


    return (; lw_radiation=lw_radiation_train, grads, loss, metrics, opt_state, grad_sum)
end